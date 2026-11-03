"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Regular polygon initialization + Simulated Annealing with worst-triangle targeting.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, None
        
    if hull_area < 1e-10:
        return 0.0, None

    n = len(pts)
    min_area = float('inf')
    worst_idx = None
    
    # Vectorized computation for speed
    # We can compute all areas at once for N=13
    # Indices for all triples
    i, j, k = np.indices((n, n, n))
    # Mask for i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    worst_idx = (i[min_idx], j[min_idx], k[min_idx])
    
    score = min_area / hull_area
    return score, worst_idx

def _compute_score(pts):
    """
    Wrapper for score only.
    """
    score, _ = _compute_score_and_worst(pts)
    return score

def _initialize():
    """
    Initialize points on a regular polygon.
    This is a strong starting point for Heilbronn-type problems.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s hard kill)
    
    # 1. Initialization
    pts = _initialize()
    current_score, _ = _compute_score_and_worst(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # 2. Simulated Annealing
    # Parameters
    T = 0.05  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    # We will perform many iterations. Each iteration is fast (vectorized area calc).
    # N=13, 286 triangles. Vectorized check is ~microseconds.
    # We can afford ~100k-500k iterations in 110s if Python overhead is low.
    # Let's aim for a high iteration count but check time frequently.
    
    rng = np.random.default_rng(42)
    
    # Precompute indices for efficiency if we were doing it manually, but vectorized is fine.
    
    while time.time() < deadline:
        # Check if we should stop cooling
        if T < min_T:
            break
            
        # Generate a perturbation
        # Strategy: Pick a random point and jitter it.
        # Alternatively, target the worst triangle.
        # Let's try a mix: 70% random point, 30% worst triangle point.
        
        current_score, worst_idx = _compute_score_and_worst(pts)
        
        if worst_idx is not None and rng.random() < 0.3:
            # Pick a point from the worst triangle
            candidate_idx = rng.choice(worst_idx)
        else:
            candidate_idx = rng.integers(0, N)
        
        # Jitter magnitude proportional to temperature
        jitter = rng.normal(0, T, size=2)
        
        new_pts = pts.copy()
        new_pts[candidate_idx] += jitter
        
        # Optional: Keep points within a reasonable bounding box to prevent explosion
        # Since the score is scale-invariant (ratio), we don't strictly need to bound,
        # but huge coordinates can cause numerical issues or convex hull failures.
        # Let's normalize new_pts to unit hull area? No, that's expensive.
        # Just clamp to [-10, 10] roughly.
        new_pts = np.clip(new_pts, -10, 10)
        
        new_score, _ = _compute_score_and_worst(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            # Improvement
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Worse, accept with probability exp((new - current) / T)
            # Note: we are maximizing, so delta = new - current is negative.
            delta = new_score - current_score
            if delta == 0:
                accept_prob = 1.0
            else:
                # Avoid division by zero or overflow
                exponent = delta / T
                if exponent < -700:
                    accept_prob = 0.0
                else:
                    accept_prob = np.exp(exponent)
            
            if rng.random() < accept_prob:
                pts = new_pts
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        
        # If T drops too low, we might get stuck. 
        # We can implement a "reheat" if no improvement for a long time, 
        # but simple cooling is often enough for this landscape if T starts high enough.
        # Given the non-smoothness, a higher initial T and slower cooling helps.
        
    # 3. Local Polish (Nelder-Mead) on the best point found
    # This helps refine the position precisely.
    if time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        
        # Flatten points for Nelder-Mead
        x0 = best_pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 2)
            s, _ = _compute_score_and_worst(p)
            return -s
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_score:
                best_pts = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
