import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute which triples involve each point for fast local updates
POINT_TRIPLES = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
    POINT_TRIPLES[i].append(idx)
    POINT_TRIPLES[j].append(idx)
    POINT_TRIPLES[k].append(idx)
POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.05):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_area - temperature * log_sum
    
    return soft_min / hull_area

def _generate_initializer_regular():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _generate_initializer_two_ring(n_out=7, n_in=7, r_in=0.6, phi_rel=0.0):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _generate_initializer_jittered(noise_scale=0.1, seed=0):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + rng.normal(0, noise_scale, N)
    # Ensure radii are positive
    radii = np.abs(radii)
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return pts

def _generate_initializer_radial_alt(seed=0):
    """Alternating radii to break symmetry"""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Alternate between 1.0 and 1.0 + delta
    delta = rng.uniform(0.05, 0.2)
    radii = np.ones(N)
    radii[1::2] = 1.0 + delta
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return pts

def _polish_with_annealing(points, time_limit=5.0):
    """
    Polishes points using L-BFGS-B on a soft-min surrogate with annealing temperature.
    """
    start_time = time.time()
    pts = points.copy()
    best_score = _calculate_score(pts)
    best_pts = pts.copy()
    
    # Temperature schedule: Start high to find basin, lower to refine
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() - start_time > time_limit:
            break
            
        def neg_soft_score_flat(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp)
        
        try:
            res = minimize(neg_soft_score_flat, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':100, 'ftol':1e-10, 'gtol':1e-6, 'maxls':50})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                # If it didn't improve, keep previous best pts for next iteration
                pts = best_pts.copy()
                
        except Exception:
            pass

    return best_pts, best_score

def _coordinate_descent_polish(points, time_limit=2.0):
    """
    Iteratively adjust each point to improve the minimum triangle area.
    Uses fast local updates for triples involving the moved point.
    """
    start_time = time.time()
    pts = points.copy()
    
    # Precompute current areas
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                tri_idx = POINT_TRIPLES[i]
                best_move = None
                best_min_area = min_area
                p_i_curr = pts[i]
                
                # Try 8 directions + center (no move)
                # We only want to move if it improves
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_p = p_i_curr + np.array([dx, dy])
                        new_areas = areas.copy()
                        
                        for t_idx in tri_idx:
                            i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                            p_a = pts[i_t] if i_t != i else new_p
                            p_b = pts[j_t] if j_t != i else new_p
                            p_c = pts[k_t] if k_t != i else new_p
                            
                            v1_t = p_b - p_a
                            v2_t = p_c - p_a
                            cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                            new_areas[t_idx] = 0.5 * np.abs(cross_t)
                        
                        curr_min = np.min(new_areas)
                        if curr_min > best_min_area:
                            best_min_area = curr_min
                            best_move = new_p
                        
                        # Check opposite
                        new_p2 = p_i_curr - np.array([dx, dy])
                        new_areas2 = areas.copy()
                        for t_idx in tri_idx:
                            i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                            p_a = pts[i_t] if i_t != i else new_p2
                            p_b = pts[j_t] if j_t != i else new_p2
                            p_c = pts[k_t] if k_t != i else new_p2
                            
                            v1_t = p_b - p_a
                            v2_t = p_c - p_a
                            cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                            new_areas2[t_idx] = 0.5 * np.abs(cross_t)
                        
                        curr_min2 = np.min(new_areas2)
                        if curr_min2 > best_min_area:
                            best_min_area = curr_min2
                            best_move = new_p2
                
                if best_move is not None:
                    pts[i] = best_move
                    improved = True
                    
                    # Recompute all areas
                    p_i = pts[TRIPLE_INDICES[:, 0]]
                    p_j = pts[TRIPLE_INDICES[:, 1]]
                    p_k = pts[TRIPLE_INDICES[:, 2]]
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    min_area = np.min(areas)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # List of initializers: (name, points)
    initializers = []
    
    # 1. Regular Polygon
    initializers.append(("reg", _generate_initializer_regular()))
    
    # 2. Two-Ring (7-7)
    initializers.append(("2ring_77", _generate_initializer_two_ring(7, 7, 0.6, 0.0)))
    initializers.append(("2ring_77_off", _generate_initializer_two_ring(7, 7, 0.6, np.pi/7)))
    
    # 3. Jittered Regular (different seeds)
    for seed in [1, 2, 3]:
        initializers.append((f"jitter_{seed}", _generate_initializer_jittered(0.1, seed)))
        
    # 4. Radial Alternating
    for seed in [10, 11]:
        initializers.append((f"radial_{seed}", _generate_initializer_radial_alt(seed)))

    # Process each initializer
    for name, pts in initializers:
        if time.time() > deadline - 10:
            break
            
        # Polish with annealing
        polish_time = min(10.0, deadline - time.time() - 5)
        if polish_time < 2.0:
            break
            
        polished_pts, polished_score = _polish_with_annealing(pts, polish_time)
        try_update(polished_score, polished_pts)
        
        # If we have time, do a final coordinate descent on the best point found so far
        # We'll do this in a final pass to save time per initializer
    
    # Final intensive polish on the global best
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        # Split time between another annealing pass and coordinate descent
        anneal_time = remaining * 0.5
        cd_time = remaining * 0.5
        
        # Try one more annealing pass with slightly different params if time permits
        if anneal_time > 1.0:
            # Use a smaller starting temperature to refine
            final_pts, final_score = _polish_with_annealing(best_points, anneal_time)
            try_update(final_score, final_pts)
            
        # Coordinate descent for exact min
        if cd_time > 0.5:
            final_pts = _coordinate_descent_polish(best_points, cd_time)
            final_score = _calculate_score(final_pts)
            try_update(final_score, final_pts)

    if best_points is None:
        best_points = _generate_initializer_regular()
        best_score = _calculate_score(best_points)

    return best_points
