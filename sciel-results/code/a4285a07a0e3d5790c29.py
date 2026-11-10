"""
Minimize logarithmic energy of 150 points on the unit sphere.
E = sum_{i<j} log(1/|x_i - x_j|)  (natural log, chordal distances)

Strategy (combined):
  - Core: custom Riemannian L-BFGS directly on the sphere (no projection artifacts).
  - Diversify: spherical-coordinate L-BFGS-B restarts to escape basins.
  - Intermediate polish: adaptive tangent gradient descent.
  - Final polish: exact Newton / quasi-Newton local solve on the tangent space
    (high-precision, tight tolerance) to drive the local minimum as tight as possible.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 8.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _retract(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _energy_and_grad(X, eps=1e-14):
    """Vectorized energy and Euclidean gradient. Returns (E, G)."""
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    E = -0.5 * float(np.sum(log_d))
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    S1 = inv @ X
    S2 = (inv.sum(axis=1, keepdims=True)) * X
    G = S2 - S1
    return E, G


def _project_tangent(G, X):
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
    n = len(v)
    q = v.copy()
    alphas = np.zeros(m)
    for k in range(m - 1, -1, -1):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        alphas[k] = rho * float(np.dot(s, q))
        q = q - alphas[k] * y
    r = np.zeros(n)
    if y_list:
        yy = float(np.dot(y_list[-1], y_list[-1]))
        if yy > 1e-20:
            gamma = float(np.dot(s_list[-1], y_list[-1])) / yy
            r = gamma * q
    for k in range(m):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        beta = rho * float(np.dot(y, r))
        r = r + (alphas[k] - beta) * s
    return q + r


def _lbfgs_sphere(X0, time_budget, m=15, max_iter=5000, tol=1e-10):
    """Riemannian L-BFGS on the sphere with retraction and strong Wolfe line search."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    while _time_left() > 2.0 and it < max_iter:
        p_flat = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p_flat.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-13:
            break

        gnorm = float(np.linalg.norm(flat))
        if gnorm < tol:
            break

        c1 = 1e-4
        c2 = 0.9
        slope = float(np.dot(flat, p_flat))
        if slope >= 0:
            p = -g
            p = _project_tangent(p, X)
            p_flat = p.ravel()
            slope = float(np.dot(flat, p_flat))
            if slope >= 0:
                break

        alpha = 1.0
        Xn, En, Gn, gn = X, E, G, g
        for _ in range(60):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        if En >= E:
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        dX = (Xn - X).ravel()
        dy = gn.ravel() - flat
        sy = float(np.dot(dX, dy))
        if sy > 1e-14:
            s_list.append(dX)
            y_list.append(dy)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        X = Xn
        E = En
        G = Gn
        g = gn
        flat = g.ravel()
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    return _retract(v)


def _structured_grid():
    n_lat, n_lon = 12, 13
    points = []
    for i in range(n_lat):
        theta = (i + 0.5) * math.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * math.pi / n_lon
            points.append([math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi),
                           math.cos(theta)])
    while len(points) < N:
        rng = np.random.default_rng(len(points) * 7919)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        points.append(v)
    return _retract(np.array(points[:N]))


def _perturb(X, seed, sigma):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape) * sigma
    return _retract(X + noise)


# --- Spherical-coordinate diversifying optimizer (from inspirations) ---
def _cartesian_to_spherical(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    return np.column_stack([theta, phi])


def _spherical_to_cartesian(params):
    theta, phi = params[:, 0], params[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _energy_grad_spherical(params):
    X = _spherical_to_cartesian(params)
    diff = X[:, None, :] - X[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    e = -np.sum(np.log(dist[mask]))
    inv = 1.0 / dist_sq
    np.fill_diagonal(inv, 0.0)
    grad_cart = -np.sum(diff * inv[:, :, None], axis=1)
    theta, phi = params[:, 0], params[:, 1]
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    gtheta = -sp * st * grad_cart[:, 0] + sp * ct * grad_cart[:, 1]
    gphi = cp * ct * grad_cart[:, 0] + cp * st * grad_cart[:, 1] - sp * grad_cart[:, 2]
    gtheta[sp < 1e-6] = 0.0
    return e, np.column_stack([gtheta, gphi]).flatten()


def _optimize_spherical(X_init):
    params = _cartesian_to_spherical(X_init).flatten()

    def fun(x):
        e, _ = _energy_grad_spherical(x.reshape(N, 2))
        return e

    def jac(x):
        _, g = _energy_grad_spherical(x.reshape(N, 2))
        return g

    try:
        res = minimize(fun, params, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
        X = _spherical_to_cartesian(res.x.reshape(N, 2))
        X = _retract(X)
        return X, _energy_and_grad(X)[0]
    except Exception:
        return X_init, _energy_and_grad(X_init)[0]


# --- Adaptive tangent gradient-descent polish (from inspiration 1) ---
def _gd_polish(X0, time_budget, lr_init=1e-4, min_lr=1e-9):
    X = _retract(X0.copy())
    E, _ = _energy_and_grad(X)
    best_X, best_E = X.copy(), E
    lr = lr_init
    patience = 0
    end = time.time() + time_budget
    while time.time() < end and _time_left() > 2.0:
        _, G = _energy_and_grad(X)
        g = _project_tangent(G, X)
        Xn = _retract(X - lr * g)
        En, _ = _energy_and_grad(Xn)
        if En < best_E - 1e-15:
            best_X, best_E = Xn.copy(), En
            patience = 0
            lr = min(lr * 1.03, 0.05)
        else:
            patience += 1
            lr = max(lr * 0.9, min_lr)
        X = Xn
        if patience > 120:
            break
    return best_X, best_E


# --- Final Newton / exact local solve on the tangent space ---
def _hessian_tangent(X):
    """Hessian of the tangent-projected energy (block matrix H_{(i,a),(j,b)}).
    H = sum_{k} w_k (delta_ij I - x_i x_j^T) with w_k = 1/|x_i - x_k|^2.
    Returns H as a 450x450 matrix (flattened)."""
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-14)
    W = 1.0 / d2  # W[i,k] = 1/|x_i - x_k|^2
    # For each i: sum_k W[i,k] (delta_ij I - x_i x_j^T)
    # H_i = I * (sum_k W[i,k]) - x_i (sum_k W[i,k] x_k)^T - (sum_k W[i,k] x_k) x_i^T
    #      = I * S_i - 2 x_i (W_i x)^T
    S = W.sum(axis=1)  # (N,)
    Wx = W @ X  # (N,3)
    # Build 3x3 blocks
    H = np.zeros((N * 3, N * 3))
    for i in range(N):
        xi = X[i]
        block = S[i] * np.eye(3) - 2.0 * np.outer(xi, Wx[i])
        H[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = block
    return H


def _newton_polish(X0, time_budget, max_iter=60):
    """Newton / damped Newton local solve on the tangent space."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    best_X, best_E = X.copy(), E
    end = time.time() + time_budget
    it = 0
    while time.time() < end and _time_left() > 2.0 and it < max_iter:
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-12:
            break
        try:
            H = _hessian_tangent(X)
            # Solve H p = -g on tangent; add small ridge for stability.
            rhs = -g.ravel()
            lam = 1e-8
            try:
                p_flat = np.linalg.solve(H + lam * np.eye(N * 3), rhs)
            except np.linalg.LinAlgError:
                p_flat = np.linalg.lstsq(H + lam * np.eye(N * 3), rhs, rcond=None)[0]
            p = p_flat.reshape(N, 3)
            p = _project_tangent(p, X)
            pn = float(np.linalg.norm(p))
            if pn < 1e-14:
                break
            # Damped Newton line search
            slope = float(np.dot(g.ravel(), p.ravel()))
            alpha = 1.0
            Xn, En, Gnn = X, E, G
            gn = g
            improved = False
            for _ in range(40):
                Xn = _retract(X + alpha * p)
                En, Gnn = _energy_and_grad(Xn)
                gn = _project_tangent(Gnn, Xn)
                if En <= E + 1e-4 * alpha * slope:
                    improved = True
                    break
                alpha *= 0.5
            if not improved or En >= E:
                # Fall back to steepest descent step
                p = -g
                p = _project_tangent(p, X)
                alpha = 1e-3
                Xn = _retract(X + alpha * p)
                En, Gnn = _energy_and_grad(Xn)
                gn = _project_tangent(Gnn, Xn)
                if En >= E:
                    break
            X, E, G, g = Xn, En, Gnn, gn
            if E < best_E:
                best_E, best_X = E, X.copy()
            it += 1
        except Exception:
            break
    return best_X, best_E


def _validate(X):
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    diff = X[:, None, :] - X[None, :, :]
    dd = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dd, 2.0)
    if float(dd.min()) < 1e-9:
        return False
    return True


def construct():
    best_X = _fibonacci_spiral()
    best_E = _energy_and_grad(best_X)[0]

    # Phase 1: Riemannian L-BFGS from diverse starts
    starts = []
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0]:
        starts.append(_fibonacci_spiral(offset))
    starts.append(_structured_grid())
    for s in [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5000, 6000]:
        starts.append(_random_points(s))

    for start in starts:
        if _time_left() > 12.0:
            X, E = _lbfgs_sphere(start, 15.0)
            if E < best_E:
                best_E, best_X = E, X

    # Phase 2: Spherical-coordinate diversifying restarts around the best
    if _time_left() > 25.0:
        for s in [11, 22, 33, 44]:
            if _time_left() < 6.0:
                break
            start = _perturb(best_X, s * 7919, 0.15)
            X, E = _optimize_spherical(start)
            if E < best_E:
                best_E, best_X = E, X
            # Re-polish with Riemannian L-BFGS to remove spherical-param artifacts
            if _time_left() > 15.0:
                X2, E2 = _lbfgs_sphere(X, 12.0)
                if E2 < best_E:
                    best_E, best_X = E2, X2

    # Phase 3: Local perturbations to escape shallow local minima
    for sigma in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        if _time_left() > 8.0:
            for trial in range(4):
                if _time_left() < 4.0:
                    break
                start = _perturb(best_X, trial * 1000 + int(sigma * 10000), sigma)
                X, E = _lbfgs_sphere(start, 10.0)
                if E < best_E:
                    best_E, best_X = E, X

    # Phase 4: Adaptive gradient-descent polish (cheap, stabilizes)
    if _time_left() > 15.0:
        X, E = _gd_polish(best_X, 10.0)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 5: High-precision Riemannian polish
    if _time_left() > 18.0:
        X, E = _lbfgs_sphere(best_X, 12.0, m=20, max_iter=5000, tol=1e-12)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 6: Final Newton / exact local solve (tightest local minimum)
    if _time_left() > 12.0:
        X, E = _newton_polish(best_X, 10.0, max_iter=60)
        if E < best_E:
            best_E, best_X = E, X
        # One more ultra-tight Riemannian pass after Newton
        if _time_left() > 8.0:
            X2, E2 = _lbfgs_sphere(best_X, 6.0, m=25, max_iter=3000, tol=1e-14)
            if E2 < best_E:
                best_E, best_X = E2, X2

    best_X = _retract(best_X)
    if not _validate(best_X):
        best_X = _fibonacci_spiral()
    return best_X
