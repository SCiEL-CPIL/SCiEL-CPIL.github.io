import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 135.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    Uses log-sum-exp for the min to make it differentiable.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    vals = -beta * areas
    v_max = np.max(vals)
    if v_max == -np.inf:
        return -np.inf
        
    log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _neg_soft_score(params, beta):
    """Negative soft score for minimization."""
    pts = params.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _local_search_worst_triangles(pts, max_iters=200, tol=1e-10):
    """
    Direct local search that targets the worst-performing triangles.
    For each iteration, find the minimum area triangle, then try to move
    one of its vertices to increase that area while not decreasing other areas.
    """
    pts = np.array(pts, dtype=np.float64)
    best_score = _score_points(pts)
    
    for it in range(max_iters):
        areas = _compute_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Get the three points of the worst triangle
        i, j, k = _triple_i[min_idx], _triple_j[min_idx], _triple_k[min_idx]
        
        # Try small perturbations of each of the three points
        improved = False
        for idx in [i, j, k]:
            orig = pts[idx].copy()
            
            # Try moving in several directions
            for dx in [-0.02, -0.01, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0.01, 0.02]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[idx] = orig + np.array([dx, dy])
                    
                    new_areas = _compute_areas(new_pts)
                    new_min = np.min(new_areas)
                    old_min = min_area
                    
                    # Accept if the minimum area increases
                    if new_min > old_min + tol:
                        pts = new_pts
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller steps
            for step in [0.005, 0.002, 0.001]:
                for idx in [i, j, k]:
                    orig = pts[idx].copy()
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        dx = step * np.cos(angle)
                        dy = step * np.sin(angle)
                        new_pts = pts.copy()
                        new_pts[idx] = orig + np.array([dx, dy])
                        
                        new_areas = _compute_areas(new_pts)
                        new_min = np.min(new_areas)
                        
                        if new_min > min_area + tol:
                            pts = new_pts
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    final_score = _score_points(pts)
    if final_score > best_score:
        best_score = final_score
        return pts, final_score
    else:
        return pts, final_score

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. Random
    rng = np.random.RandomState(42)
    p4 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p4)
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_initial_guesses()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
    
    # 2. Simulated Annealing with Local Refinement
    T0 = 0.05
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 20000
    
    current_params = best_params.copy()
    current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta=50.0)
    
    rng = np.random.RandomState(123)
    
    # Track acceptance rate for adaptive step size
    accept_count = 0
    total_count = 0
    step_size = 0.1
    
    for i in range(max_iters):
        # Time check
        if (time.time() - t_start) > (DEADLINE - 15.0):
            break
            
        T = T0 * (cooling_rate ** i)
        beta = 20.0 + 100.0 * (1.0 - T / T0)
        
        perturbation = rng.normal(0, step_size, size=(2 * N))
        neighbor_params = current_params + perturbation
        neighbor_params = np.clip(neighbor_params, -1.0, 1.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, beta)
        
        if neighbor_soft_score > current_soft_score:
            accept = True
        else:
            delta = current_soft_score - neighbor_soft_score
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
        total_count += 1
        if accept:
            accept_count += 1
            current_params = neighbor_params
            current_soft_score = neighbor_soft_score
            
            # Occasionally perform local refinement on the soft score
            if i % 50 == 0 and i > 0:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, beta),
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res.fun < _neg_soft_score(current_params, beta):
                        current_params = res.x
                        current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta)
                except Exception:
                    pass
            
            # Update global best with hard score
            if i % 10 == 0:
                hard_s = _score_points(current_params.reshape(N, 2))
                if hard_s > best_score:
                    best_score = hard_s
                    best_params = current_params.copy()
        
        # Adaptive step size based on acceptance rate
        if total_count > 100:
            accept_rate = accept_count / total_count
            if accept_rate > 0.7:
                step_size *= 1.1
            elif accept_rate < 0.3:
                step_size *= 0.9
            step_size = np.clip(step_size, 0.01, 0.5)
            accept_count = 0
            total_count = 0

    # 3. Direct local search on worst triangles - this is the key improvement
    if (time.time() - t_start) < (DEADLINE - 10.0):
        pts_ls, score_ls = _local_search_worst_triangles(best_params.reshape(N, 2), max_iters=300)
        if score_ls > best_score:
            best_score = score_ls
            best_params = pts_ls.flatten().copy()
    
    # 4. Final Local Refinement on Hard Score
    starts_to_refine = [best_params]
    
    if not np.allclose(best_params, current_params):
        starts_to_refine.append(current_params)
        
    for p in inits:
        starts_to_refine.append(p)
        
    unique_starts = []
    for s in starts_to_refine:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
            
    for start in unique_starts:
        if (time.time() - t_start) > (DEADLINE - 5.0):
            break
            
        try:
            beta_high = 500.0
            res = minimize(
                lambda x: _neg_soft_score(x, beta_high),
                start,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12}
            )
            score2 = _score_points(res.x.reshape(N, 2))
            if score2 > best_score:
                best_score = score2
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Also try local search on worst triangles from this refined point
        if (time.time() - t_start) < (DEADLINE - 3.0):
            pts_ls2, score_ls2 = _local_search_worst_triangles(res.x.reshape(N, 2), max_iters=100)
            if score_ls2 > best_score:
                best_score = score_ls2
                best_params = pts_ls2.flatten().copy()
    
    return best_params.reshape(N, 2)
