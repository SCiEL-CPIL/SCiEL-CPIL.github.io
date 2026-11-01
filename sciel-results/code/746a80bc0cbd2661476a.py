import numpy as np
import scipy.spatial as spat
import time
import itertools
from scipy.optimize import minimize
import sys

N = 14
DEADLINE = 145  # Safety margin before 150s kill

def _calc_min_tri_area(points):
    """
    Compute the minimum triangle area over all triples.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Pre-compute combinations of indices for n=14 (364 triples)
    # This is fast enough to do every time or cache, but doing it every time is safer for shape changes
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(angle_offset, angle_offset + 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _generate_ring_config(radii, angles_offsets, n_per_ring):
    """
    Generate points from a set of concentric rings.
    radii: list of radii for each ring
    angles_offsets: list of angular offsets for each ring
    n_per_ring: list of number of points per ring
    """
    points = []
    for r, ang_off, n_pts in zip(radii, angles_offsets, n_per_ring):
        if n_pts > 0:
            pts = _regular_polygon(n_pts, radius=r, angle_offset=ang_off)
            points.append(pts)
    if not points:
        return np.zeros((N, 2))
    return np.vstack(points)

def _neg_score(points):
    """Objective function for minimization (negative score)."""
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # 1. Initial Heuristic Configurations
    # Strategy: Try various ring structures (1, 2, 3, 4 rings)
    # The optimal solution for "max min triangle area" often involves symmetry.
    # Common structures:
    # - 1 ring: Regular 14-gon
    # - 2 rings: 7+7, 6+8, 5+9 etc.
    # - 3 rings: 5+4+5, 6+4+4 etc.
    
    initial_configs = []
    
    # Regular 14-gon
    initial_configs.append(_regular_polygon(N))
    
    # 2 Rings
    for n1 in range(4, 11):
        n2 = N - n1
        if n2 < 3: continue
        # Ratio of radii
        for ratio in [0.5, 0.6, 0.7, 0.8]:
            for rot1 in [0.0, np.pi/N]:
                for rot2 in [0.0, np.pi/N]:
                    pts = _generate_ring_config([ratio, 1.0], [rot1, rot2], [n1, n2])
                    initial_configs.append(pts)
                    
    # 3 Rings
    for n1 in range(3, 8):
        for n2 in range(3, 8):
            n3 = N - n1 - n2
            if n3 < 3: continue
            for r1 in [0.4, 0.5]:
                for r2 in [0.7, 0.8]:
                    pts = _generate_ring_config([r1, r2, 1.0], [0.0, np.pi/N, 0.0], [n1, n2, n3])
                    initial_configs.append(pts)

    # Evaluate initial configs
    for init_pts in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        # Normalize to unit hull area
        hull_a = _hull_area(init_pts)
        if hull_a < 1e-10:
            continue
        init_pts = init_pts / np.sqrt(hull_a)
        score = _score(init_pts)
        update_best(init_pts, score)
        
    # 2. Local Refinement using Nelder-Mead on Full Coordinates
    # Nelder-Mead is derivative-free and robust to non-smoothness (kinks in min function)
    # compared to L-BFGS. We run it from multiple diverse starting points.
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15 and best_points is not None:
        # Select top 5 diverse initial points
        # We just picked the best one, but we can also try refining other good ones
        # Let's keep a list of candidate starts
        candidate_starts = []
        
        # Add the current best
        if best_points is not None:
            candidate_starts.append(best_points.copy())
            
        # Add some other promising initial configs (e.g., regular polygon, best 2-ring)
        # We'll just re-evaluate a few specific ones to ensure diversity
        specific_starts = [
            _regular_polygon(N),
            _generate_ring_config([0.7, 1.0], [0.0, np.pi/N], [7, 7]),
            _generate_ring_config([0.5, 0.8, 1.0], [0.0, np.pi/N, 0.0], [5, 4, 5])
        ]
        
        for s in specific_starts:
            hull_a = _hull_area(s)
            if hull_a > 1e-10:
                s_norm = s / np.sqrt(hull_a)
                candidate_starts.append(s_norm)
                
        # Limit number of starts to save time
        max_starts = 5
        for start_pts in candidate_starts[:max_starts]:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
                
            try:
                # Nelder-Mead parameters
                # initial_simplex: default is based on x0
                # maxiter: limit to prevent hanging
                res = minimize(
                    _neg_score,
                    start_pts.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if res.fun < 0: # Check if it found a valid point
                    res_pts = res.x.reshape(-1, 2)
                    hull_a = _hull_area(res_pts)
                    if hull_a > 1e-10:
                        res_pts = res_pts / np.sqrt(hull_a)
                        res_score = _score(res_pts)
                        update_best(res_pts, res_score)
                        
            except Exception:
                pass

    # 3. Parameterized Ring Optimization
    # If the full coordinate search didn't jump significantly, try optimizing the ring parameters directly.
    # This reduces dimensionality from 28 to ~6-10, allowing for more thorough search.
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        
        def optimize_ring_structure(n1, n2, n3):
            """
            Optimize radii and angles for a 3-ring structure.
            Parameters: [r1, r2, theta1, theta2]
            """
            def ring_obj(params):
                r1, r2, t1, t2 = params
                # Ensure valid radii
                r1 = np.clip(r1, 0.1, 0.9)
                r2 = np.clip(r2, 0.2, 1.0)
                
                pts = _generate_ring_config([r1, r2, 1.0], [t1, t2, 0.0], [n1, n2, n3])
                hull_a = _hull_area(pts)
                if hull_a < 1e-10:
                    return 1e10
                pts = pts / np.sqrt(hull_a)
                return -_score(pts)
            
            # Initial guess
            x0 = [0.5, 0.8, 0.0, np.pi/N]
            
            try:
                res = minimize(
                    ring_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True}
                )
                
                r1, r2, t1, t2 = res.x
                r1 = np.clip(r1, 0.1, 0.9)
                r2 = np.clip(r2, 0.2, 1.0)
                pts = _generate_ring_config([r1, r2, 1.0], [t1, t2, 0.0], [n1, n2, n3])
                hull_a = _hull_area(pts)
                if hull_a > 1e-10:
                    pts = pts / np.sqrt(hull_a)
                    score = _score(pts)
                    update_best(pts, score)
                    
            except Exception:
                pass

        # Try a few specific ring splits
        # 5-4-5, 6-4-4, 7-3-4, 4-5-5
        ring_splits = [(5, 4, 5), (6, 4, 4), (7, 3, 4), (4, 5, 5), (3, 5, 6)]
        
        for split in ring_splits:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            optimize_ring_structure(*split)

    # 4. Final Polish: Small perturbations around best point
    # Sometimes Nelder-Mead stops slightly before the optimum due to tolerances.
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2 and best_points is not None:
        current_pts = best_points.copy()
        current_score = _score(current_pts)
        
        rng = np.random.RandomState(42)
        step_size = 0.01
        
        for i in range(100):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 1:
                break
                
            new_pts = current_pts.copy()
            idx = rng.randint(N)
            noise = rng.normal(0, step_size, 2)
            new_pts[idx] += noise
            
            hull_a = _hull_area(new_pts)
            if hull_a < 1e-10:
                continue
            
            new_pts = new_pts / np.sqrt(hull_a)
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                update_best(current_pts, current_score)
            else:
                # Reduce step size if no improvement
                if i % 20 == 19:
                    step_size *= 0.5

    if best_points is None:
        best_points = initial_configs[0]
        
    return np.array(best_points)
