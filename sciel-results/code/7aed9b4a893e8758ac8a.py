import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # safety margin before 150s kill


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


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
    for _ in range(200):
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


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _physics_relax(xyz, time_limit=3.0, grow_rate=0.002):
    """
    Force-directed physics relaxation with Hertzian contact model.
    Uses non-linear repulsive forces proportional to penetration^(3/2).
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.003
    damping = 0.98
    k_contact = 500.0  # Hertzian stiffness
    
    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise Hertzian repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                
                # Hertzian contact: F = k * delta^(3/2)
                delta = r[i] + r[j] - dist
                if delta > 0:
                    force = k_contact * (delta ** 1.5)
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Wall repulsion (Hertzian-like)
        for i in range(N):
            # Left wall
            delta = r[i] - x[i]
            if delta > 0:
                force = k_contact * (delta ** 1.5)
                fx[i] += force
            
            # Right wall
            delta = (x[i] + r[i]) - 1.0
            if delta > 0:
                force = k_contact * (delta ** 1.5)
                fx[i] -= force
            
            # Bottom wall
            delta = r[i] - y[i]
            if delta > 0:
                force = k_contact * (delta ** 1.5)
                fy[i] += force
            
            # Top wall
            delta = (y[i] + r[i]) - 1.0
            if delta > 0:
                force = k_contact * (delta ** 1.5)
                fy[i] -= force
        
        # Update positions with velocity damping
        x += fx * dt
        y += fy * dt
        
        # Keep within bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # Update xyz
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check if valid and improve best
        valid_xyz = _enforce_validity(xyz)
        current_score = _score_valid(valid_xyz)
        if current_score > best_score:
            best_score = current_score
            best_xyz = valid_xyz.copy()
    
    return best_xyz


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
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
            
            # Phase 1: L-BFGS-B with softplus penalty method
            try:
                def obj(p):
                    v, _ = _objective_grad(p, 1000.0)
                    return v
                
                def grad(p):
                    _, g = _objective_grad(p, 1000.0)
                    return g
                
                res = minimize(
                    obj,
                    p0.flatten(),
                    method='L-BFGS-B',
                    jac=grad,
                    bounds=bounds_list,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                p = _enforce_validity(res.x.reshape(N, 3))
                s = _score_valid(p)
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception:
                continue
            
            # Phase 2: Physics-based relaxation with Hertzian contacts
            if time.time() > deadline - 15:
                break
            
            if best_params is not None:
                try:
                    p_physics = _physics_relax(
                        best_params, 
                        time_limit=min(3.0, max(0.1, deadline - time.time() - 5))
                    )
                    s_physics = _score_valid(p_physics)
                    if s_physics > best_score:
                        best_score = s_physics
                        best_params = p_physics.copy()
                except Exception:
                    continue
            
            # Phase 3: Analytic gradient polish
            if time.time() > deadline - 8:
                break
            
            if best_params is not None:
                try:
                    p_polished = _polish_analytic(
                        best_params,
                        time_budget=min(4.0, max(0.1, deadline - time.time() - 3))
                    )
                    s_polished = _score_valid(p_polished)
                    if s_polished > best_score:
                        best_score = s_polished
                        best_params = p_polished.copy()
                except Exception:
                    continue

    # If we didn't find any valid solution, use a fallback
    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42)).reshape(N, 3)

    # Final refinement: try to increase radii slightly if possible
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        
        p = best_params.copy()
        p[:, 2] *= 1.001
        p_new = _enforce_validity(p)
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_params = p_new
        else:
            break

    # Return in the required format: (x, y, radius)
    result = best_params.copy()
    return result
