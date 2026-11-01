"""Packing 200 equal circles in a unit square via smooth soft-min gradient ascent."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 150s wall-clock


def _true_score(P):
    """Exact objective: min(pairwise dist, wall clearance)."""
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -1.0
    D = P[:, None, :] - P[None, :, :]
    D = np.linalg.norm(D, axis=2)
    np.fill_diagonal(D, np.inf)
    r_pair = D.min()
    L = np.abs(P)
    r_wall = (0.5 - L.max(axis=1)).min()
    return float(min(r_pair, r_wall))


def _softmin_and_grad(vals, beta):
    """Soft-min of vals and its gradient (vectorized)."""
    m = vals.min()
    z = vals - m
    logsum = m + np.log(np.sum(np.exp(-beta * z)))
    w = np.exp(-beta * (vals - logsum))
    w /= w.sum()
    return logsum, w


def _grad(P, beta):
    D = P[:, None, :] - P[None, :, :]
    dist = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dist, np.inf)
    _, wp = _softmin_and_grad(dist.ravel(), beta)
    wp = wp.reshape(N, N)
    G = -2.0 * (wp @ D)
    # wall terms
    L = np.abs(P)
    wm = (0.5 - L).ravel()
    _, ww = _softmin_and_grad(wm, beta)
    ww = ww.reshape(N, 2)
    G += -2.0 * ww * np.sign(P)
    return G


def _optimize(P, beta, steps, base_lr, deadline):
    P = P.astype(np.float64, copy=True)
    best = _true_score(P)
    bestP = P.copy()
    lr = base_lr
    for _ in range(steps):
        if time.time() > deadline:
            break
        G = _grad(P, beta)
        n = np.linalg.norm(G)
        if n < 1e-16:
            break
        Pn = P + lr * G / n
        s = _true_score(Pn)
        if s > best + 1e-14:
            P = Pn
            best = s
            bestP = Pn.copy()
            lr = min(lr * 1.05, 3e-3)
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    return bestP, best


def _slsqp_polish(P, deadline):
    """Final polish using SLSQP on the active constraints."""
    P = np.asarray(P, dtype=np.float64).copy()
    if time.time() > deadline:
        return P, _true_score(P)
    
    # Find active pairs and wall constraints
    D = P[:, None, :] - P[None, :, :]
    dist = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dist, np.inf)
    r = _true_score(P)
    
    # Active pairs: distance close to 2r
    active_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if dist[i, j] < 2*r + 1e-4:
                active_pairs.append((i, j))
    
    # Active walls
    L = np.abs(P)
    active_walls = []
    for i in range(N):
        for k in range(2):
            if 0.5 - L[i, k] < r + 1e-4:
                active_walls.append((i, k))
    
    if not active_pairs and not active_walls:
        return P, _true_score(P)
    
    def objective(x):
        P2 = x.reshape(N, 2)
        return -_true_score(P2)
    
    def gradient(x):
        P2 = x.reshape(N, 2)
        return -_grad(P2, beta=2000.0).ravel()
    
    # Constraints: pairwise distance >= 2r, wall clearance >= r
    constraints = []
    for (i, j) in active_pairs[:50]:  # limit to avoid too many constraints
        def make_pair_constraint(i, j):
            def c(x):
                P2 = x.reshape(N, 2)
                return np.linalg.norm(P2[i] - P2[j]) - 2*r
            return {'type': 'ineq', 'fun': c}
        constraints.append(make_pair_constraint(i, j))
    
    for (i, k) in active_walls[:50]:
        def make_wall_constraint(i, k):
            def c(x):
                P2 = x.reshape(N, 2)
                return 0.5 - np.abs(P2[i, k]) - r
            return {'type': 'ineq', 'fun': c}
        constraints.append(make_wall_constraint(i, k))
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(objective, P.ravel(), method='SLSQP', jac=gradient,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-12})
        if res.success and _true_score(res.x.reshape(N, 2)) > _true_score(P):
            P = res.x.reshape(N, 2)
    except Exception:
        pass
    
    return P, _true_score(P)


def _random_init(seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(-0.5, 0.5, size=(N, 2))
    return P


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _jittered_grid_init(seed):
    P = _grid_init()
    rng = np.random.default_rng(seed)
    P = P + rng.normal(0, 0.005, size=P.shape)
    return P


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0

    bestP = _grid_init()
    best = _true_score(bestP)

    # Phase 1: many cheap random-restart soft-min gradient ascents.
    beta = 1000.0
    seed = 0
    while time.time() < _DEADLINE - 30.0:
        seed += 1
        if seed % 4 == 0:
            P0 = _grid_init()
        elif seed % 2 == 0:
            P0 = _jittered_grid_init(seed)
        else:
            P0 = _random_init(seed)
        P1, s = _optimize(P0, beta, steps=400, base_lr=5e-4, deadline=_DEADLINE - 30.0)
        if s > best:
            best = s
            bestP = P1.copy()
        if time.time() > _DEADLINE - 30.0:
            break

    # Phase 2: high-precision polish on the best configuration.
    while time.time() < _DEADLINE - 15.0:
        P1, s = _optimize(bestP, beta=2000.0, steps=600, base_lr=1e-5, deadline=_DEADLINE - 15.0)
        if s > best:
            best = s
            bestP = P1.copy()
        else:
            # polish converged
            break

    # Phase 3: SLSQP polish on active constraints
    if time.time() < _DEADLINE - 5.0:
        P1, s = _slsqp_polish(bestP, _DEADLINE - 5.0)
        if s > best:
            best = s
            bestP = P1.copy()

    # Final safety: ensure finite and in-bounds.
    bestP = np.clip(bestP, -0.5, 0.5)
    return bestP
