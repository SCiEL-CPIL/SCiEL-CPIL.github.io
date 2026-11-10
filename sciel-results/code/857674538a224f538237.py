"""Rewritten from scratch: physics-based L-BFGS-B circle packing in [-0.5,0.5]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 135.0  # safety margin below the 150 s kill


def _min_dist(X):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.maximum(d2, 0.0)).min()


def _wall(X):
    return 0.5 - np.abs(X).max()


def _score(X):
    return min(_min_dist(X) * 0.5, _wall(X))


def _energy(X, r, alpha):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    D = r / d
    E = (D ** 2).sum()
    w = np.maximum(0.0, 0.5 - np.abs(X))
    E += alpha * np.sum(1.0 / (w + 1e-9) ** 2)
    return E


def _energy_grad(X, r, alpha):
    n = X.shape[0]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.maximum(d2, 0.0))
    D = r / d
    D.fill_diagonal(0.0)
    diff = X[:, None, :] - X[None, :, :]
    G = np.zeros_like(X)
    np.einsum('ij,ijk->ik', D ** 2, diff, out=G)
    w = 0.5 - np.abs(X)
    G += alpha * (2.0 / (w + 1e-9) ** 3) * np.sign(X)
    return G


def _grid_init(n):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].copy()


def _epigraph_obj(vars, alpha):
    """
    Epigraph formulation: maximize t subject to:
      t <= d_ij / 2 for all i<j
      t <= 0.5 - |x_i| for all i,k
    We minimize -t + penalty terms.
    
    vars = [X (N*2), t]
    """
    n = N
    X = vars[:2*n].reshape(n, 2)
    t = vars[2*n]
    
    # Pairwise distances
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.maximum(d2, 0.0))
    
    # Smooth min of d/2 over pairs using log-sum-exp with large beta
    # min(d/2) approx = -(1/beta) * log(sum(exp(-beta * d/2)))
    # But we want to enforce t <= min(d/2), so we penalize when t > d/2
    # Use soft violation: sum of max(0, t - d/2)^2 approximated smoothly
    
    # For pairs, compute violation = max(0, t - d/2)
    # Smooth approximation: 0.5 * (v + sqrt(v^2 + eps)) where v = t - d/2
    eps = 1e-6
    v_pair = t - d * 0.5
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    v_pair = v_pair[iu]
    # Smooth max(0, v)^2
    smooth_viol_pair = 0.5 * (v_pair + np.sqrt(v_pair**2 + eps)) ** 2
    E_pair = np.sum(smooth_viol_pair)
    
    # Wall constraints: t <= 0.5 - |x_ik|
    w = 0.5 - np.abs(X)
    v_wall = t - w
    smooth_viol_wall = 0.5 * (v_wall + np.sqrt(v_wall**2 + eps)) ** 2
    E_wall = alpha * np.sum(smooth_viol_wall)
    
    # Objective: minimize -t + violations
    obj = -t + E_pair + E_wall
    return obj


def _epigraph_grad(vars, alpha):
    n = N
    X = vars[:2*n].reshape(n, 2)
    t = vars[2*n]
    
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.maximum(d2, 0.0))
    diff = X[:, None, :] - X[None, :, :]
    
    iu = np.triu_indices(n, 1)
    v_pair = t - d[iu] * 0.5
    eps = 1e-6
    # d/dv of 0.5*(v + sqrt(v^2+eps))^2 = (v + sqrt(v^2+eps)) * (1 + v/sqrt(v^2+eps))
    sqrt_term = np.sqrt(v_pair**2 + eps)
    dsmooth_dv = (v_pair + sqrt_term) * (1.0 + v_pair / sqrt_term)
    
    # Gradient w.r.t. t from pair violations
    G_t = -1.0 + np.sum(dsmooth_dv * (-0.5))  # dv_pair/dt = 1, but dsmooth/dv already accounts
    # Actually: d(E_pair)/dt = sum(dsmooth_dv * 1)
    G_t = -1.0 + np.sum(dsmooth_dv)
    
    # Gradient w.r.t. X from pair violations
    # d(E_pair)/dX_i = sum_j dsmooth_dv_ij * d(v_ij)/dX_i
    # v_ij = t - 0.5*d_ij, d(v_ij)/dX_i = -0.5 * d_ij_vec / d_ij
    G_X = np.zeros_like(X)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = d[i, j]
        if d_ij < 1e-12:
            continue
        # d(v_ij)/dX_i = -0.5 * (X_i - X_j) / d_ij
        grad_v = -0.5 * (X[i] - X[j]) / d_ij
        G_X[i] += dsmooth_dv[idx] * grad_v
        G_X[j] -= dsmooth_dv[idx] * grad_v
    
    # Wall violations
    w = 0.5 - np.abs(X)
    v_wall = t - w
    sqrt_w = np.sqrt(v_wall**2 + eps)
    dsmooth_dw = (v_wall + sqrt_w) * (1.0 + v_wall / sqrt_w)
    
    # d(E_wall)/dt = sum(dsmooth_dw * 1)
    G_t += alpha * np.sum(dsmooth_dw)
    
    # d(E_wall)/dX_ik: v_wall_ik = t - (0.5 - |x_ik|)
    # d(v_wall_ik)/dX_ik = -d(0.5 - |x_ik|)/dX_ik = sign(X_ik)
    G_X += alpha * dsmooth_dw * np.sign(X)
    
    grad = np.concatenate([G_X.flatten(), [G_t]])
    return grad


def _optimize(X0, r0, alpha, budget):
    X = X0.copy()
    cur = X
    for _ in range(3):
        t0 = time.time()
        res = minimize(_energy, X, jac=_energy_grad, method='L-BFGS-B',
                       args=(r0, alpha),
                       options={'maxiter': 400, 'maxfun': 20000,
                                'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None and np.all(np.isfinite(res.x)):
            X = res.x
            cur = X
        if time.time() - t0 > budget or time.time() > DEADLINE:
            break
    return cur


def _optimize_epigraph(X0, t0, alpha, budget):
    """Optimize using epigraph formulation with auxiliary variable t."""
    n = N
    vars0 = np.concatenate([X0.flatten(), [t0]])
    
    t_start = time.time()
    res = minimize(_epigraph_obj, vars0, jac=_epigraph_grad, method='L-BFGS-B',
                   args=(alpha,),
                   options={'maxiter': 500, 'maxfun': 30000,
                            'maxls': 50, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if res.x is not None and np.all(np.isfinite(res.x)):
        X = res.x[:2*n].reshape(n, 2)
        # Clip to ensure validity
        X = np.clip(X, -0.5, 0.5)
        return X, res.x[2*n]
    
    return X0, t0


def construct():
    start = time.time()
    r_best = 0.0
    X_best = _grid_init(N)
    r_best = _score(X_best)

    # Coarse anneal: start from a wide radius so forces spread particles out,
    # then shrink the target radius to tighten the packing.
    alphas = [40.0, 10.0, 4.0, 1.5, 0.6, 0.25]
    radii = [0.055, 0.050, 0.045, 0.042, 0.039, 0.0375]
    X = _grid_init(N)

    for r0, alpha in zip(radii, alphas):
        if time.time() > DEADLINE:
            break
        X = _optimize(X, r0, alpha, budget=18.0)
        s = _score(X)
        if s > r_best:
            r_best = s
            X_best = X.copy()

    # Use epigraph formulation for refinement
    if time.time() < DEADLINE - 20.0:
        t_init = _score(X)
        X_ep, t_ep = _optimize_epigraph(X, t_init, 0.5, budget=30.0)
        s_ep = _score(X_ep)
        if s_ep > r_best:
            r_best = s_ep
            X_best = X_ep.copy()
            X = X_ep
    
    # Final local polish near the known optimum radius.
    if time.time() < DEADLINE - 15.0:
        for r0 in (0.0368, 0.0366, 0.0364):
            if time.time() > DEADLINE - 12.0:
                break
            X = _optimize(X, r0, 0.15, budget=10.0)
            s = _score(X)
            if s > r_best:
                r_best = s
                X_best = X.copy()

    # High-precision final polish with very tight tolerances
    if time.time() < DEADLINE - 5.0:
        r_polish = _score(X_best)
        X_polish = X_best.copy()
        res = minimize(_energy, X_polish, jac=_energy_grad, method='L-BFGS-B',
                       args=(r_polish, 0.1),
                       options={'maxiter': 1000, 'maxfun': 50000,
                                'maxls': 60, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.x is not None and np.all(np.isfinite(res.x)):
            X_polish = np.clip(res.x, -0.5, 0.5)
            s_polish = _score(X_polish)
            if s_polish > r_best:
                r_best = s_polish
                X_best = X_polish.copy()

    # Safety: ensure a valid finite array inside the square.
    X_best = np.clip(X_best, -0.5, 0.5)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    return X_best
