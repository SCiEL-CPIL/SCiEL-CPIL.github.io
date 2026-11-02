import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(n):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _get_contact_edges(p, margin=0.05):
    """Get Delaunay contact edges plus nearby pairs."""
    points = p[:, :2]
    edges = set()
    
    try:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    except Exception:
        pass
    
    # Add pairs that are close (potential contacts)
    r = p[:, 2]
    for i in range(N):
        for j in range(i+1, N):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] + margin:
                edges.add((i, j))
    
    return sorted(edges)


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback to grid if Voronoi fails
        return _grid_initial(seed)
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        # Distance to nearest neighbor
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        
        # Distance to walls
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        
        # Initial radius is half the min distance, with a small safety factor
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    # Small jitter to break symmetry
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
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
    radii = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _corner_weighted_initial(seed):
    """Initialize with bias toward corners and edges for better packing."""
    rng = np.random.RandomState(seed)
    points = np.zeros((N, 2))
    
    # Place some circles near corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(4):
        points[i] = corners[i] + rng.uniform(-0.05, 0.05, size=2)
    
    # Place some along edges
    for i in range(4, 12):
        if i % 4 == 0:
            points[i] = [rng.uniform(0.1, 0.9), 0.1]
        elif i % 4 == 1:
            points[i] = [rng.uniform(0.1, 0.9), 0.9]
        elif i % 4 == 2:
            points[i] = [0.1, rng.uniform(0.1, 0.9)]
        else:
            points[i] = [0.9, rng.uniform(0.1, 0.9)]
    
    # Remaining in center region
    for i in range(12, N):
        points[i] = rng.uniform(0.2, 0.8, size=2)
    
    # Estimate radii based on local density
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _random_initial(seed):
    """Simple random initialization with radius estimation."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


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


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(100):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle by ov (or split if equal)
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _contact_objective(params, edges, penalty=500.0):
    """Objective using only contact edges for speed."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations - only contact edges
    pair = 0.0
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        ov = (r[i] + r[j]) - dist
        if ov > 0:
            pair += ov

    return score + penalty * (pair + wall)


def _contact_grad(params, edges, penalty=500.0):
    """Gradient using only contact edges."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    g = np.zeros_like(params)

    g[2::3] = -1.0

    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Contact edge contributions
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-10:
            dist = 1e-10
        ov = (r[i] + r[j]) - dist
        if ov > 0:
            ux = dx / dist
            uy = dy / dist
            g[3*i] += penalty * (-ux)
            g[3*j] += penalty * (+ux)
            g[3*i+1] += penalty * (-uy)
            g[3*j+1] += penalty * (+uy)
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _contact_optimize(p, time_limit=8.0, penalty_start=100.0, penalty_growth=1.5, max_penalty=5000.0):
    """
    Iterative contact-based optimization with adaptive penalty.
    Uses Delaunay to identify relevant constraints, making each solve much faster.
    """
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    penalty = penalty_start
    
    while time.time() - start_time < time_limit and penalty < max_penalty:
        # Get current contact edges
        edges = _get_contact_edges(best_p, margin=0.03)
        
        # Optimize with current contact graph
        try:
            res = minimize(
                lambda params: _contact_objective(params, edges, penalty),
                best_p.flatten(),
                method='L-BFGS-B',
                jac=lambda params: _contact_grad(params, edges, penalty),
                bounds=bounds_list,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            p_new = _enforce_validity(res.x.reshape(N, 3))
            s_new = _score_valid(p_new)
            
            if s_new > best_score:
                best_score = s_new
                best_p = p_new
            else:
                # No improvement, increase penalty
                penalty *= penalty_growth
                continue
                
        except Exception:
            penalty *= penalty_growth
            continue
        
        # Check if we're still making progress
        if time.time() - start_time > time_limit * 0.8:
            break
    
    return best_p, best_score


def _local_optimize_with_contacts(p, time_limit=8.0, seed=42):
    """
    Local optimization using contact graph approach.
    Multiple restarts with different penalties and perturbations.
    """
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    rng = np.random.RandomState(seed)
    
    # First pass: direct contact optimization
    p_opt, s_opt = _contact_optimize(p, time_limit=min(time_limit, 5.0))
    if s_opt > best_score:
        best_score = s_opt
        best_p = p_opt
    
    # Second pass: perturb and re-optimize
    if time.time() - start_time < time_limit - 2:
        for _ in range(3):
            if time.time() - start_time > time_limit - 1:
                break
            
            # Small perturbation
            p_pert = best_p.copy()
            p_pert[:, :2] += rng.normal(0, 0.01, size=(N, 2))
            p_pert[:, 2] += rng.normal(0, 0.005, size=N)
            p_pert = _enforce_validity(p_pert)
            
            p_opt, s_opt = _contact_optimize(p_pert, time_limit=min(3.0, max(0.5, time_limit - (time.time() - start_time))))
            if s_opt > best_score:
                best_score = s_opt
                best_p = p_opt
    
    return best_p, best_score


def _shake_and_refine(p, steps=3000, time_limit=15.0, seed=42):
    """
    Refine a valid configuration using random perturbations with contact-based optimization.
    """
    current_score = _score_valid(p)
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    
    start = time.time()
    sigma = 0.02
    
    for _ in range(steps):
        if time.time() - start > time_limit:
            break
            
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.5, size=N)
        
        p_new = _enforce_validity(p_new)
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_p = p_new
        
        # Use contact-based optimization instead of slow SLSQP
        if _ % 200 == 0 and time.time() - start < time_limit - 5:
            p_try, s_try = _local_optimize_with_contacts(
                p_new, 
                time_limit=min(4.0, max(0.5, time_limit - (time.time() - start) - 3)),
                seed=seed + _
            )
            if s_try > best_score:
                best_score = s_try
                best_p = p_try
    
    return best_p, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
        lambda s: _corner_weighted_initial(s),
        lambda s: _random_initial(s),
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
            
            # First, enforce validity to get a safe starting point
            p0 = _enforce_validity(x0)
            
            # Phase 1: Contact-based optimization (fast)
            try:
                p, s = _local_optimize_with_contacts(
                    p0,
                    time_limit=min(6.0, max(0.5, deadline - time.time() - 5)),
                    seed=seed
                )
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception:
                continue
            
            # Phase 2: Additional refinement if time permits
            if time.time() > deadline - 15:
                break
            
            if best_params is not None and time.time() < deadline - 20:
                try:
                    p_refined, s_refined = _local_optimize_with_contacts(
                        best_params, 
                        time_limit=min(5.0, max(0.5, deadline - time.time() - 8)),
                        seed=seed + 1000
                    )
                    if s_refined > best_score:
                        best_score = s_refined
                        best_params = p_refined.copy()
                except Exception:
                    continue

    # If we didn't find any valid solution, use a fallback
    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42)).reshape(N, 3)

    # Final refinement: shake and refine with SLSQP
    if time.time() < deadline - 5:
        try:
            p_final, s_final = _shake_and_refine(
                best_params,
                steps=5000,
                time_limit=min(20.0, max(0.1, deadline - time.time() - 2)),
                seed=42
            )
            if s_final > best_score:
                best_score = s_final
                best_params = p_final
        except Exception:
            pass

    # Final refinement: try to increase radii slightly if possible
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        
        p = best_params.copy()
        # Try to increase all radii by a tiny amount
        p[:, 2] *= 1.001
        
        # Enforce validity (this will shrink radii if needed)
        p_new = _enforce_validity(p)
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_params = p_new
        else:
            # Can't improve, break
            break

    # Return in the required format: (x, y, radius)
    result = best_params.copy()
    return result
