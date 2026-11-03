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

def _worst_triangles(pts, k=5):
    """Return indices of the k smallest triangles."""
    areas = _compute_areas(pts)
    n = min(k, len(areas))
    idx = np.argpartition(areas, n)[:n]
    return idx, areas[idx]

def _move_point_toward_centroid(pts, idx, tri_idx, strength=0.3):
    """
    Move point idx toward the centroid of triangle tri_idx by 'strength' fraction.
    This is a heuristic to increase the area of the worst triangle.
    """
    i, j, k = _triple_i[tri_idx], _triple_j[tri_idx], _triple_k[tri_idx]
    centroid = (pts[i] + pts[j] + pts[k]) / 3.0
    pts[idx] = pts[idx] + strength * (centroid - pts[idx])
    return pts

def _local_search_worst_triangles(pts, max_iter=200, deadline=None):
    """
    Iteratively identify worst triangles and move their vertices toward the triangle centroid.
    This directly targets the bottleneck.
    """
    pts = pts.copy()
    best_score = _score_points(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and (time.time() - deadline) < 10.0:
            break
            
        # Find worst triangles
        worst_idx, worst_areas = _worst_triangles(pts, k=3)
        
        if worst_areas[0] < 1e-15:
            # Degenerate, try to fix
            for wi in worst_idx:
                i, j, k = _triple_i[wi], _triple_j[wi], _triple_k[wi]
                # Move one of the points slightly
                pts[i] += 0.01 * np.random.randn(2)
            continue
        
        # For each worst triangle, try moving each of its 3 vertices toward centroid
        improved = False
        for wi in worst_idx:
            i, j, k = _triple_i[wi], _triple_j[wi], _triple_k[wi]
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            
            for vertex in [i, j, k]:
                # Try moving this vertex toward centroid with various strengths
                for strength in [0.1, 0.2, 0.3, 0.5]:
                    trial = pts.copy()
                    trial[vertex] = trial[vertex] + strength * (centroid - trial[vertex])
                    trial_score = _score_points(trial)
                    if trial_score > best_score + 1e-10:
                        pts = trial
                        best_score = trial_score
                        best_pts = pts.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller perturbations
            for wi in worst_idx:
                i, j, k = _triple_i[wi], _triple_j[wi], _triple_k[wi]
                centroid = (pts[i] + pts[j] + pts[k]) / 3.0
                for vertex in [i, j, k]:
                    trial = pts.copy()
                    # Small random nudge
                    trial[vertex] += 0.005 * np.random.randn(2)
                    trial_score = _score_points(trial)
                    if trial_score > best_score + 1e-10:
                        pts = trial
                        best_score = trial_score
                        best_pts = pts.copy()
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
    return best_pts, best_score

def _coordinate_descent(pts, max_iter=100, deadline=None):
    """
    Coordinate descent: optimize one coordinate at a time using golden section or fine grid.
    """
    pts = pts.copy()
    best_score = _score_points(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and (time.time() - deadline) < 10.0:
            break
            
        improved = False
        for dim in range(2 * N):
            point_idx = dim // 2
            coord_idx = dim % 2
            
            # Try a few steps in each direction
            for step in [0.01, 0.02, 0.05, 0.1]:
                for direction in [1, -1]:
                    trial = pts.copy()
                    trial[point_idx, coord_idx] += direction * step
                    trial_score = _score_points(trial)
                    if trial_score > best_score + 1e-10:
                        pts = trial
                        best_score = trial_score
                        best_pts = pts.copy()
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_pts, best_score

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
    
    # 5. 2 centers + 11 outer (slightly offset centers)
    p5 = np.zeros((N, 2))
    p5[0] = [-0.1, 0]
    p5[1] = [0.1, 0]
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    p5[2:, 0] = np.cos(angles)
    p5[2:, 1] = np.sin(angles)
    inits.append(p5.flatten())
    
    return inits

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    # Stable logsumexp calculation for soft-min
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
            
    # 2. Phase 1: Quick SA to find a good starting region
    T0 = 0.05
    T_end = 1e-5
    cooling_rate = 0.998
    max_sa_iters = 5000
    
    current_params = best_params.copy()
    rng = np.random.RandomState(123)
    
    for i in range(max_sa_iters):
        if (time.time() - t_start) > (DEADLINE - 60.0):
            break
            
        T = T0 * (cooling_rate ** i)
        beta = 20.0 + 100.0 * (1.0 - T / T0)
        
        perturbation = rng.normal(0, 0.08, size=(2 * N))
        neighbor_params = current_params + perturbation
        neighbor_params = np.clip(neighbor_params, -1.0, 1.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, beta)
        current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta)
        
        if neighbor_soft_score > current_soft_score:
            accept = True
        else:
            delta = current_soft_score - neighbor_soft_score
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
        if accept:
            current_params = neighbor_params
            hard_s = _score_points(current_params.reshape(N, 2))
            if hard_s > best_score:
                best_score = hard_s
                best_params = current_params.copy()
                
    # 3. Phase 2: Direct worst-triangle local search (the key structural change)
    # This directly targets the bottleneck triangles
    start_pts = best_params.reshape(N, 2)
    improved_pts, improved_score = _local_search_worst_triangles(
        start_pts, max_iter=300, deadline=t_start
    )
    if improved_score > best_score:
        best_score = improved_score
        best_params = improved_pts.flatten()
        
    # 4. Phase 3: Coordinate descent for fine-tuning
    start_pts = best_params.reshape(N, 2)
    cd_pts, cd_score = _coordinate_descent(start_pts, max_iter=200, deadline=t_start)
    if cd_score > best_score:
        best_score = cd_score
        best_params = cd_pts.flatten()
        
    # 5. Phase 4: More worst-triangle local search from the refined point
    start_pts = best_params.reshape(N, 2)
    improved_pts, improved_score = _local_search_worst_triangles(
        start_pts, max_iter=300, deadline=t_start
    )
    if improved_score > best_score:
        best_score = improved_score
        best_params = improved_pts.flatten()
        
    # 6. Phase 5: Another coordinate descent pass
    start_pts = best_params.reshape(N, 2)
    cd_pts, cd_score = _coordinate_descent(start_pts, max_iter=200, deadline=t_start)
    if cd_score > best_score:
        best_score = cd_score
        best_params = cd_pts.flatten()
        
    # 7. Final: Try local search from a few diverse starting points if time allows
    if (time.time() - t_start) < (DEADLINE - 20.0):
        for p in inits[:3]:
            if (time.time() - t_start) > (DEADLINE - 10.0):
                break
            pts = p.reshape(N, 2)
            ls_pts, ls_score = _local_search_worst_triangles(pts, max_iter=100, deadline=t_start)
            if ls_score > best_score:
                best_score = ls_score
                best_params = ls_pts.flatten()
                
    return best_params.reshape(N, 2)
