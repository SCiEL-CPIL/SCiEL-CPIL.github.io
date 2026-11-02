import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 26
DEADLINE = 145.0


def _validate(p):
    """Check if configuration is valid. Returns (is_valid, min_violation)."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Wall violations
    wall_viol = (np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) +
                 np.maximum(0, r - y) + np.maximum(0, r - (1 - y))).max()
    
    # Pairwise violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    pair_viol = np.maximum(0, (r[iu[0]] + r[iu[1]]) - dist).max()
    
    min_viol = max(wall_viol, pair_viol)
    return min_viol < 1e-9, min_viol


def _fix_boundaries(p):
    """Fix boundary violations by shrinking radii."""
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0, 1)
    p[:, 1] = np.clip(p[:, 1], 0, 1)
    p[:, 2] = np.clip(p[:, 2], 1e-10, 0.5)
    
    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]),
                      np.minimum(p[:, 1], 1 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-10))
    
    # Resolve overlaps iteratively
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-12
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i, j = iu[0][k], iu[1][k]
            ov = overlap[k]
            # Shrink smaller circle
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-10)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    p[j, 2] -= min(rem, p[j, 2] - 1e-10)
            else:
                shrink = min(ov, p[j, 2] - 1e-10)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    p[i, 2] -= min(rem, p[i, 2] - 1e-10)
        wall = np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]),
                          np.minimum(p[:, 1], 1 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-10))
    
    return p


def _softplus(x, beta=100):
    """Smooth approximation of max(0, x)."""
    return np.log1p(np.exp(beta * x)) / beta


def _objective_penalty(params, penalty=500.0):
    """Penalty-based objective: minimize -sum(r) + penalty * violations."""
    p = params.reshape(N, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    
    score = -np.sum(r)
    
    # Wall violations
    wx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
    wy = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    wall = np.sum(wx) + np.sum(wy)
    
    # Pairwise overlaps
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    overlap = np.maximum(0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)
    
    return score + penalty * (pair + wall)


def _objective_grad(params, penalty=500.0):
    """Gradient of penalty objective."""
    p = params.reshape(N, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    g = np.zeros_like(params)
    
    # d(-sum r)/dr = -1
    g[2::3] = -1.0
    
    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] -= penalty
            g[3*i+2] += penalty
        if r[i] - (1 - x[i]) > 0:
            g[3*i] += penalty
            g[3*i+2] += penalty
        if r[i] - y[i] > 0:
            g[3*i+1] -= penalty
            g[3*i+2] += penalty
        if r[i] - (1 - y[i]) > 0:
            g[3*i+1] += penalty
            g[3*i+2] += penalty
    
    # Pairwise
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax, ay = iu[0][active], iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i, j = ax[k], ay[k]
            g[3*i] -= penalty * adx[k]
            g[3*j] += penalty * adx[k]
            g[3*i+1] -= penalty * ady[k]
            g[3*j+1] += penalty * ady[k]
            g[3*i+2] += penalty
            g[3*j+2] += penalty
    
    return g


def _delaunay_contact_graph(p):
    """Get contact edges from Delaunay triangulation."""
    try:
        delaunay = Delaunay(p[:, :2])
        edges = set()
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []


def _get_contact_edges(p, tol=1e-4):
    """Get edges where circles are nearly touching."""
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    contact = np.abs((r[iu[0]] + r[iu[1]]) - dist) < tol
    return list(zip(iu[0][contact], iu[1][contact]))


def _initial_hex(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(8):
        for j in range(8):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    radii = np.full(N, 0.04)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _initial_grid(seed):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _initial_corner_weighted(seed):
    """Initialize with circles weighted toward corners."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    # Push some circles toward corners
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    for i in range(min(4, N)):
        points[i] = corners[i] + rng.uniform(-0.05, 0.05, 2)
    
    points = np.clip(points, 0.05, 0.95)
    radii = np.full(N, 0.04)
    
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.06, size=N)
    
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _score_valid(p):
    return np.sum(p[:, 2])


def _slsqp_constrained(p, time_limit=15.0, seed=42):
    """SLSQP optimization with all constraints."""
    best_p = p.copy()
    best_score = _score_valid(best_p)
    start = time.time()
    
    def obj(params):
        return -np.sum(params[2::3])
    
    def obj_grad(params):
        g = np.zeros_like(params)
        g[2::3] = -1.0
        return g
    
    cons = []
    
    # Wall constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2],
                     'jac': lambda p, i=i: np.concatenate([np.zeros(3*i), [1, 0, -1], np.zeros(3*(N-i)-3)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2],
                     'jac': lambda p, i=i: np.concatenate([np.zeros(3*i), [-1, 0, -1], np.zeros(3*(N-i)-3)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
                     'jac': lambda p, i=i: np.concatenate([np.zeros(3*i), [0, 1, -1], np.zeros(3*(N-i)-3)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2],
                     'jac': lambda p, i=i: np.concatenate([np.zeros(3*i), [0, -1, -1], np.zeros(3*(N-i)-3)])})
    
    # Pair constraints
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        def pair_fun(p, i=i, j=j):
            dx = p[3*i] - p[3*j]
            dy = p[3*i+1] - p[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - p[3*i+2] - p[3*j+2]
        def pair_jac(p, i=i, j=j):
            g = np.zeros_like(p)
            dx = p[3*i] - p[3*j]
            dy = p[3*i+1] - p[3*j+1]
            dist = max(np.sqrt(dx*dx + dy*dy), 1e-10)
            g[3*i] = dx / dist
            g[3*j] = -dx / dist
            g[3*i+1] = dy / dist
            g[3*j+1] = -dy / dist
            g[3*i+2] = -1
            g[3*j+2] = -1
            return g
        cons.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})
    
    lo, hi = _bounds()
    bounds = list(zip(lo, hi))
    
    try:
        res = minimize(obj, best_p.flatten(), method='SLSQP', jac=obj_grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            p_new = res.x.reshape(N, 3)
            p_new = _fix_boundaries(p_new)
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new
    except Exception:
        pass
    
    return best_p


def _penalty_optimize(p, time_limit=10.0, seed=42):
    """Penalty-based optimization."""
    best_p = p.copy()
    best_score = _score_valid(best_p)
    start = time.time()
    
    # Try different penalty values
    for penalty in [100, 500, 1000, 2000]:
        if time.time() > start + time_limit:
            break
        try:
            res = minimize(_objective_penalty, best_p.flatten(), method='BFGS',
                           jac=lambda x: _objective_grad(x, penalty),
                           options={'maxiter': 100, 'gtol': 1e-8})
            if res.success:
                p_new = res.x.reshape(N, 3)
                p_new = _fix_boundaries(p_new)
                s_new = _score_valid(p_new)
                if s_new > best_score:
                    best_score = s_new
                    best_p = p_new
        except Exception:
            pass
    
    return best_p


def _shake_and_refine(p, time_limit=15.0, seed=42):
    """Stochastic shake and refine."""
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    sigma = 0.015
    
    for _ in range(10000):
        if time.time() > start + time_limit:
            break
        
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.3, size=N)
        p_new = _fix_boundaries(p_new)
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_p = p_new
    
    return best_p


def _high_precision_polish(p, time_limit=10.0, seed=42):
    """Final high-precision polish using trust-constr."""
    best_p = p.copy()
    best_score = _score_valid(best_p)
    start = time.time()
    
    def obj(params):
        return -np.sum(params[2::3])
    
    def obj_grad(params):
        g = np.zeros_like(params)
        g[2::3] = -1.0
        return g
    
    cons = []
    
    # Wall constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
    
    # Pair constraints
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j:
                     np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]})
    
    lo, hi = _bounds()
    bounds = list(zip(lo, hi))
    
    try:
        res = minimize(obj, best_p.flatten(), method='trust-constr',
                       jac=obj_grad, bounds=bounds, constraints=cons,
                       options={'maxiter': 100, 'gtol': 1e-10})
        if res.success:
            p_new = res.x.reshape(N, 3)
            p_new = _fix_boundaries(p_new)
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new
    except Exception:
        pass
    
    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_score = -np.inf
    
    initializers = [
        lambda s: _initial_hex(s),
        lambda s: _initial_grid(s),
        lambda s: _initial_corner_weighted(s),
        lambda s: _initial_random(s),
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            
            x0 = init_fn(seed)
            x0 = _fix_boundaries(x0.reshape(N, 3)).flatten()
            
            # Penalty optimization
            if time.time() < deadline - 15:
                p1 = _penalty_optimize(x0.reshape(N, 3), time_limit=8.0, seed=seed)
                s1 = _score_valid(p1)
                if s1 > best_score:
                    best_score = s1
                    best_params = p1.copy()
            
            # SLSQP on best so far
            if best_params is not None and time.time() < deadline - 20:
                p2 = _slsqp_constrained(best_params, time_limit=12.0, seed=seed+100)
                s2 = _score_valid(p2)
                if s2 > best_score:
                    best_score = s2
                    best_params = p2.copy()
            
            # Shake
            if best_params is not None and time.time() < deadline - 15:
                p3 = _shake_and_refine(best_params, time_limit=10.0, seed=seed+200)
                s3 = _score_valid(p3)
                if s3 > best_score:
                    best_score = s3
                    best_params = p3.copy()
    
    # Final polish
    if best_params is not None and time.time() < deadline - 5:
        p_final = _high_precision_polish(best_params, time_limit=8.0, seed=999)
        s_final = _score_valid(p_final)
        if s_final > best_score:
            best_score = s_final
            best_params = p_final.copy()
    
    if best_params is None:
        best_params = _fix_boundaries(_initial_grid(42).reshape(N, 3))
    
    return best_params
