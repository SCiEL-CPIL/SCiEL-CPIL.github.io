import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill

# Global time tracking
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE - _elapsed()

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
    """
    Enforce all constraints: walls and pairwise non-overlap by shrinking radii.
    This is the safety net to ensure returned solutions are valid.
    """
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

def _physics_relax(xyz, time_limit=2.0, grow_rate=0.002, seed=42):
    """
    Force-directed physics relaxation that grows radii.
    This is the primary exploration engine.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = _score_valid(xyz)
    best_xyz = xyz.copy()
    
    dt = 0.002
    damping = 0.9
    
    rng = np.random.RandomState(seed)
    
    for iteration in range(20000):
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
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Force proportional to overlap
                    force_mag = overlap * 10.0
                    fx[i] += force_mag * dx / dist
                    fy[i] += force_mag * dy / dist
                    fx[j] -= force_mag * dx / dist
                    fy[j] -= force_mag * dy / dist
        
        # Wall forces (push away if touching)
        for i in range(N):
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 10.0
            if x[i] + r[i] > 1:
                fx[i] -= (x[i] + r[i] - 1) * 10.0
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 10.0
            if y[i] + r[i] > 1:
                fy[i] -= (y[i] + r[i] - 1) * 10.0
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Damping
        # We don't store velocity in this simple version, so damping is implicit in the force calculation or we can just rely on the small dt.
        # To add simple damping, we would need to track velocity. Let's stick to simple Euler integration for now.
        
        # Clip positions to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Enforce validity strictly to prevent divergence
        current_xyz = np.column_stack((x, y, r))
        current_xyz = _enforce_validity(current_xyz)
        
        s = _score_valid(current_xyz)
        if s > best_score:
            best_score = s
            best_xyz = current_xyz.copy()
            
    return best_xyz

def _slsqp_refine(p, time_limit=5.0, seed=42):
    """
    Iterative SLSQP refinement based on contact graph.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    iu = np.triu_indices(N, k=1)
    pair_indices = list(zip(iu[0], iu[1]))
    
    for iteration in range(20):
        if time.time() > start + time_limit:
            break
        
        x = best_p[:, 0]
        y = best_p[:, 1]
        r = best_p[:, 2]
        
        # Identify active constraints (contacts)
        tol = 1e-4
        
        cons = []
        
        # Wall constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[3*i] - params[3*i+2], 
                         'jac': lambda params, i=i: _wall_jac(3*i, 3*i+2, 1.0, -1.0, params)})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: 1.0 - params[3*i] - params[3*i+2], 
                         'jac': lambda params, i=i: _wall_jac(3*i, 3*i+2, -1.0, -1.0, params)})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[3*i+1] - params[3*i+2], 
                         'jac': lambda params, i=i: _wall_jac(3*i+1, 3*i+2, 1.0, -1.0, params)})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: 1.0 - params[3*i+1] - params[3*i+2], 
                         'jac': lambda params, i=i: _wall_jac(3*i+1, 3*i+2, -1.0, -1.0, params)})
            
        # Pairwise constraints (only near-active)
        for i, j in pair_indices:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] + tol:
                cons.append({'type': 'ineq', 'fun': lambda params, i=i, j=j: _pair_dist(params, i, j), 
                             'jac': lambda params, i=i, j=j: _pair_jac(params, i, j)})

        lo, hi = _bounds()
        bounds = list(zip(lo, hi))
        
        def obj(params):
            return -np.sum(params[2::3])
            
        def obj_grad(params):
            g = np.zeros_like(params)
            g[2::3] = -1.0
            return g
            
        try:
            res = minimize(
                obj,
                best_p.flatten(),
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                p_new = res.x.reshape(N, 3)
                p_new = _enforce_validity(p_new)
                s_new = _score_valid(p_new)
                
                if s_new > best_score:
                    best_score = s_new
                    best_p = p_new
                else:
                    if iteration > 5:
                        best_p = best_p + rng.normal(0, 1e-5, size=best_p.shape)
                        best_p = _enforce_validity(best_p)
                        
        except Exception:
            pass
            
    return best_p

def _wall_jac(idx1, idx2, c1, c2, params):
    g = np.zeros_like(params)
    g[idx1] = c1
    g[idx2] = c2
    return g

def _pair_dist(params, i, j):
    dx = params[3*i] - params[3*j]
    dy = params[3*i+1] - params[3*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - params[3*i+2] - params[3*j+2]

def _pair_jac(params, i, j):
    g = np.zeros_like(params)
    dx = params[3*i] - params[3*j]
    dy = params[3*i+1] - params[3*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist < 1e-10:
        dist = 1e-10
    g[3*i] = dx / dist
    g[3*j] = -dx / dist
    g[3*i+1] = dy / dist
    g[3*j+1] = -dy / dist
    g[3*i+2] = -1.0
    g[3*j+2] = -1.0
    return g

def _analytic_polish(p, time_limit=3.0):
    """
    Final polish using L-BFGS-B with a softplus penalty for smooth gradients.
    """
    start_time = time.time()
    params = p.flatten().copy()
    best_params = params.copy()
    
    lo, hi = _bounds()
    bounds = list(zip(lo, hi))
    
    def softplus(z, eps=1e-4):
        return np.logaddexp(0, z / eps) * eps
        
    def sigmoid(z, eps=1e-4):
        return 1.0 / (1.0 + np.exp(-z / eps))
        
    def objective_grad(params, penalty_weight):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        val = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0
        
        # Boundary violations
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        sp_lx = softplus(lx)
        sg_lx = sigmoid(lx)
        grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
        grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx
        
        sp_ux = softplus(ux)
        sg_ux = sigmoid(ux)
        grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
        grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux
        
        sp_ly = softplus(ly)
        sg_ly = sigmoid(ly)
        grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
        grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly
        
        sp_uy = softplus(uy)
        sg_uy = sigmoid(uy)
        grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
        grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy
        
        val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2))
        
        # Pairwise overlaps
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        sp_ov = softplus(overlap)
        sg_ov = sigmoid(overlap)
        
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

    best_val, _ = objective_grad(best_params, 1000.0)
    
    for penalty_weight in [1000.0, 5000.0]:
        if time.time() - start_time > time_limit:
            break
            
        def obj(p):
            v, _ = objective_grad(p, penalty_weight)
            return v
            
        def grad(p):
            _, g = objective_grad(p, penalty_weight)
            return g
            
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14})
            obj_val, _ = objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _grid_initial(seed):
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

def _hex_initial(seed):
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

def _voronoi_initial(seed):
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    try:
        vor = Voronoi(points)
    except Exception:
        return _grid_initial(seed)
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

def construct():
    global _start_time
    _start_time = time.time()
    
    best_params = None
    best_score = -np.inf
    
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 11, 22]
    
    candidates = []
    
    # Phase 1: Physics Exploration
    # We run physics on multiple starts to find diverse good packings
    for seed in seeds:
        if _remaining() < 10.0: # Reserve 10s for SLSQP/Polish
            break
            
        for init_fn in initializers:
            if _remaining() < 10.0:
                break
                
            try:
                x0 = init_fn(seed)
                x0 = _enforce_validity(x0).reshape(N, 3)
                
                # Run physics relaxation
                # Allocate ~2s per physics run
                p_phys = _physics_relax(x0, time_limit=2.0, seed=seed)
                s = _score_valid(p_phys)
                
                candidates.append((s, p_phys, seed))
                
            except Exception:
                continue
                
    if not candidates:
        # Fallback
        p = _enforce_validity(_grid_initial(42)).reshape(N, 3)
        return p
        
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: SLSQP Refinement on Top Candidates
    # Take top 5 candidates and refine them with SLSQP
    top_k = 5
    for i in range(min(top_k, len(candidates))):
        if _remaining() < 5.0:
            break
        s, p, seed = candidates[i]
        
        # Run SLSQP
        p_refined = _slsqp_refine(p, time_limit=4.0, seed=seed)
        s_refined = _score_valid(p_refined)
        
        if s_refined > best_score:
            best_score = s_refined
            best_params = p_refined.copy()
            
    # If no SLSQP ran or failed, use best physics result
    if best_params is None:
        best_params = candidates[0][1]
        best_score = candidates[0][0]
        
    # Phase 3: Analytic Polish
    if _remaining() > 3.0:
        p_polished = _analytic_polish(best_params, time_limit=min(_remaining() - 1.0, 5.0))
        s_polished = _score_valid(p_polished)
        
        if s_polished > best_score:
            best_score = s_polished
            best_params = p_polished
            
    # Final safety check
    best_params = _enforce_validity(best_params)
    
    return best_params
