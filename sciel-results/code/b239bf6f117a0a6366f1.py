"""SLSQP epigraph polish with local search and multiple initializations."""
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

N = 50


def _radius_fast(pts):
    """Compute packing radius for points in [-0.5, 0.5]^2."""
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _slsqp_epigraph_polish(pts, time_limit=30.0):
    """
    SLSQP epigraph formulation: maximize t subject to all constraints >= t.
    Variables: [x_1, y_1, ..., x_N, y_N, t]
    Objective: -t (minimize)
    Constraints: pairwise distances/2 >= t, wall distances >= t
    """
    pts = pts.copy()
    r0 = _radius_fast(pts)
    if r0 <= 0:
        return pts, r0
    
    x0 = np.concatenate([pts.flatten(), [r0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    def objective(v):
        return -v[-1]
    
    def grad_objective(v):
        g = np.zeros_like(v)
        g[-1] = -1.0
        return g
    
    # Precompute pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    
    def constraints(v):
        pts_v = v[:2*N].reshape(N, 2)
        t = v[-1]
        cons = []
        # Pairwise constraints: dist(i,j)/2 - t >= 0
        for i, j in pairs:
            d = np.sqrt(np.sum((pts_v[i] - pts_v[j]) ** 2))
            cons.append(d / 2.0 - t)
        # Wall constraints: wall_dist - t >= 0
        for i in range(N):
            for k in range(2):
                cons.append(0.5 - abs(pts_v[i, k]) - t)
        return cons
    
    def jac_constraints(v):
        pts_v = v[:2*N].reshape(N, 2)
        t = v[-1]
        jac = []
        # Pairwise constraints
        for i, j in pairs:
            diff = pts_v[i] - pts_v[j]
            d = np.sqrt(np.sum(diff ** 2))
            if d < 1e-12:
                row = np.zeros(2 * N + 1)
            else:
                row = np.zeros(2 * N + 1)
                row[2*i] = diff[0] / (2 * d)
                row[2*i+1] = diff[1] / (2 * d)
                row[2*j] = -diff[0] / (2 * d)
                row[2*j+1] = -diff[1] / (2 * d)
            row[-1] = -1.0
            jac.append(row)
        # Wall constraints
        for i in range(N):
            for k in range(2):
                row = np.zeros(2 * N + 1)
                row[2*i + k] = -np.sign(pts_v[i, k]) if abs(pts_v[i, k]) > 1e-12 else 0.0
                row[-1] = -1.0
                jac.append(row)
        return np.array(jac)
    
    start = time.time()
    best_pts = pts.copy()
    best_r = r0
    
    try:
        result = minimize(
            objective,
            x0,
            jac=grad_objective,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jac_constraints}],
            options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
        )
        if result.x is not None:
            new_pts = result.x[:2*N].reshape(N, 2)
            new_pts = np.clip(new_pts, -0.499999, 0.499999)
            new_r = _radius_fast(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_r


def _local_search(centers, max_iter=2000, time_limit=10.0, step_init=0.01):
    """
    Hill-climbing local search.
    """
    centers = centers.copy()
    best_r = _radius_fast(centers)
    step_size = step_init
    no_improve = 0
    start = time.time()

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        indices = np.random.permutation(N)
        for idx in indices:
            orig = centers[idx].copy()
            wall_d = 0.5 - np.abs(orig)
            min_wall = np.min(wall_d)
            diffs = centers - orig
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[idx] = np.inf
            min_pair = np.min(dists) / 2.0
            cur_local = min(min_wall, min_pair)

            closest_idx = np.argmin(dists)
            if dists[closest_idx] / 2.0 < min_wall:
                direction = orig - centers[closest_idx]
            else:
                direction = -orig
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm

            best_local = cur_local
            best_new = orig.copy()
            for _ in range(20):
                noise = np.random.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.4
                new_c = np.clip(orig + move, -0.4999, 0.4999)
                nw = 0.5 - np.abs(new_c)
                nmin_wall = np.min(nw)
                nd = centers - new_c
                ndists = np.sqrt(np.sum(nd ** 2, axis=1))
                ndists[idx] = np.inf
                nmin_pair = np.min(ndists) / 2.0
                nlocal = min(nmin_wall, nmin_pair)
                if nlocal > best_local + 1e-12:
                    best_local = nlocal
                    best_new = new_c
            if best_local > cur_local + 1e-12:
                centers[idx] = best_new
                improved = True
        if not improved:
            no_improve += 1
            step_size *= 0.9
            if no_improve > 30:
                break
        else:
            no_improve = 0
        r = _radius_fast(centers)
        if r > best_r:
            best_r = r
    return centers, best_r


def _init_hexagonal(seed=42):
    """Hexagonal grid initialization."""
    np.random.seed(seed)
    s = 0.14
    pts = []
    for i, x in enumerate(np.arange(-0.8, 0.8, s)):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in np.arange(-0.8, 0.8, s):
            pts.append((x + offset, y))
    pts = np.array(pts)
    mask = (np.abs(pts[:, 0]) <= 0.49) & (np.abs(pts[:, 1]) <= 0.49)
    pts = pts[mask]
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()


def _init_7x7_plus(seed=123):
    """7x7 grid with some points removed to get 50."""
    np.random.seed(seed)
    gx = np.linspace(-0.45, 0.45, 7)
    gy = np.linspace(-0.45, 0.45, 7)
    pts = np.array([[x, y] for x in gx for y in gy])
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()


def _init_jittered_8x8(seed=43):
    """8x8 grid with jitter, select 50."""
    np.random.seed(seed)
    gx = np.linspace(-0.48, 0.48, 8)
    gy = np.linspace(-0.48, 0.48, 8)
    pts = np.array([[x, y] for x in gx for y in gy])
    pts = pts + np.random.normal(0, 0.01, (len(pts), 2))
    pts = np.clip(pts, -0.49, 0.49)
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()


def _init_random(seed=999):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 2))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    inits = [
        _init_hexagonal(42),
        _init_7x7_plus(123),
        _init_jittered_8x8(43),
        _init_random(999),
        _init_random(1000),
        _init_hexagonal(777),
    ]
    
    best_pts = inits[0].copy()
    best_r = _radius_fast(best_pts)
    
    # Phase 1: Local search on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        remaining = deadline - time.time()
        ls_time = min(10.0, remaining * 0.5)
        if ls_time < 2.0:
            break
        ls_c, ls_r = _local_search(init, max_iter=2000, time_limit=ls_time)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_c.copy()
    
    # Phase 2: SLSQP epigraph polish (primary optimization)
    if time.time() < deadline - 20.0:
        slsqp_c, slsqp_r = _slsqp_epigraph_polish(best_pts, time_limit=30.0)
        if slsqp_r > best_r:
            best_r = slsqp_r
            best_pts = slsqp_c.copy()
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 5.0:
        final_c, final_r = _local_search(best_pts, max_iter=1000, time_limit=10.0)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_c.copy()
    
    # Phase 4: One more SLSQP if time remains
    if time.time() < deadline - 3.0:
        slsqp_c2, slsqp_r2 = _slsqp_epigraph_polish(best_pts, time_limit=15.0)
        if slsqp_r2 > best_r:
            best_r = slsqp_r2
            best_pts = slsqp_c2.copy()
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
