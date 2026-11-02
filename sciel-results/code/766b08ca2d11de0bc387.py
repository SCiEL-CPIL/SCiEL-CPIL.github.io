"""
Heilbronn n=14: Simulated Annealing with Multi-Scale Perturbations.
Strategy:
1. Start from a regular polygon.
2. Use Simulated Annealing (SA) to explore the configuration space.
3. Perturbations are drawn from a Gaussian distribution with a scale that 
   decreases over time (cooling schedule).
4. Accept worse moves with probability exp(-delta_E / T) to escape local minima.
5. Periodically perform a local polish using Nelder-Mead to refine the current best.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START
def _min_tri_area(pts):
    """
    Calculate the minimum area of all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(pts)
    # We can optimize this further, but for N=14, O(N^3) is 2730 triangles.
    # Vectorizing the cross product is efficient.
    # i, j, k with i < j < k
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(abs(cross) * 0.5)
    return np.min(areas)

def _score(pts):
    """
    Score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    min_a = _min_tri_area(pts)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

def _regular_n_gon():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _local_polish(pts, deadline):
    """
    Perform a short Nelder-Mead optimization to refine the current point set.
    This helps SA by providing a "gradient-like" push in the best direction.
    """
    if time.time() > deadline - 2:
        return pts, _score(pts)
    
    x0 = pts.flatten()
    
    def obj(flat):
        p = flat.reshape(N, 2)
        return -_score(p)
        
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < 1e9:
            new_pts = res.x.reshape(N, 2)
            new_score = -res.fun
            return new_pts, new_score
    except Exception:
        pass
        
    return pts, _score(pts)

def construct():
    """
    Main entry point.
    Uses Simulated Annealing to maximize the minimum triangle area relative to hull area.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialize with regular polygon
    current_pts = _regular_n_gon()
    current_score = _score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.5
    T_end = 1e-6
    cooling_rate = 0.995 # Adjust based on expected iterations
    
    # Estimate max iterations
    # Each iteration: _min_tri_area (~2730 ops) + _score (Hull) 
    # Hull is the bottleneck. Let's say ~1ms per eval.
    # 145s / 0.001s = 145,000 iterations possible.
    # But we also do local polish.
    
    rng = np.random.RandomState(42)
    
    iter_count = 0
    T = T_start
    
    while time.time() < deadline - 10.0:
        iter_count += 1
        
        # 1. Generate neighbor
        # Scale perturbation by current temperature
        # As T drops, perturbations become smaller
        scale = T * 0.5 + 0.01 # Floor to avoid zero scale
        
        # Perturb all points
        noise = rng.normal(0, scale, (N, 2))
        neighbor_pts = current_pts + noise
        
        # 2. Evaluate neighbor
        neighbor_score = _score(neighbor_pts)
        
        # 3. Acceptance criterion
        delta_score = neighbor_score - current_score
        
        if delta_score > 0:
            # Always accept better
            current_pts = neighbor_pts
            current_score = neighbor_score
        else:
            # Accept worse with probability exp(-delta/T)
            # Note: delta_score is negative here
            # We want to accept if random < exp(delta_score / T)
            # Since delta_score is negative, exp(negative/positive) is < 1
            prob = np.exp(delta_score / T)
            if rng.random() < prob:
                current_pts = neighbor_pts
                current_score = neighbor_score
        
        # 4. Update best
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # 5. Cooling
        T *= cooling_rate
        
        # 6. Periodic Local Polish
        # Every 500 iterations, try to refine the current best
        if iter_count % 500 == 0 and time.time() < deadline - 20.0:
            polished_pts, polished_score = _local_polish(best_pts, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                # Reset current to polished to continue from better spot
                current_pts = best_pts.copy()
                current_score = best_score
                # Reheat slightly to escape the local minimum found by NM?
                # Or just continue cooling. Let's reheat a bit to explore around the new best.
                T = max(T, 0.1) 

    # Final Polish on the absolute best
    if time.time() < deadline - 2.0:
        final_pts, final_score = _local_polish(best_pts, deadline)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()

    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
    except Exception:
        best_pts = _regular_n_gon()
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
    
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_pts = best_pts * scale
        
    return best_pts
# EVOLVE-BLOCK-END
