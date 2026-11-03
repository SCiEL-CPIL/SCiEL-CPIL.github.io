"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Halton sequence initialization + Parallel Simulated Annealing with worst-triangle targeting.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1]^2 for a given base."""
    seq = np.zeros((n, 2))
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        for d in range(1, 3):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, d-1] = r
    return seq

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
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
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
    Initialize points using a Halton sequence, scaled to unit area.
    This provides a more uniform distribution than a regular polygon.
    """
    pts = _halton_sequence(N, 2)
    # Normalize to unit area
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area > 1e-10:
            pts /= np.sqrt(area)
    except Exception:
        pass
    return pts

def _sa_run(start_pts, deadline, seed):
    """
    Run a single Simulated Annealing chain from start_pts until deadline.
    Returns the best points found.
    """
    pts = start_pts.copy()
    current_score, _ = _compute_score_and_worst(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    T = 0.05
    cooling_rate = 0.9995
    min_T = 1e-6
    
    rng = np.random.default_rng(seed)
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        current_score, worst_idx = _compute_score_and_worst(pts)
        
        if worst_idx is not None and rng.random() < 0.3:
            candidate_idx = rng.choice(worst_idx)
        else:
            candidate_idx = rng.integers(0, N)
        
        jitter = rng.normal(0, T, size=2)
        new_pts = pts.copy()
        new_pts[candidate_idx] += jitter
        new_pts = np.clip(new_pts, -10, 10)
        
        new_score, _ = _compute_score_and_worst(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta == 0:
                accept_prob = 1.0
            else:
                exponent = delta / T
                if exponent < -700:
                    accept_prob = 0.0
                else:
                    accept_prob = np.exp(exponent)
            
            if rng.random() < accept_prob:
                pts = new_pts
                current_score = new_score
                
        T *= cooling_rate
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Generate multiple diverse starting points
    # 1. Halton sequence
    # 2. Regular polygon
    # 3. Random points
    # 4. Another Halton variant (different base)
    
    inits = []
    
    # Halton
    inits.append(_initialize())
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Random
    rng = np.random.default_rng(123)
    inits.append(rng.normal(0, 1, size=(N, 2)))
    
    # Halton base 3
    pts_h3 = _halton_sequence(N, 3)
    try:
        hull = ConvexHull(pts_h3)
        area = hull.volume
        if area > 1e-10:
            pts_h3 /= np.sqrt(area)
    except Exception:
        pass
    inits.append(pts_h3)
    
    # Run SA on each, allocating time based on available time
    # We'll run them in parallel-like fashion by time-slicing
    # Actually, we can just run them sequentially with a time slice each
    # Or, since SA is fast, we can run many iterations of the best one
    
    # Let's run a short SA on each to find the best starting point, then do a long SA on that
    # Time per init: 5s each, so 20s for 4 inits
    # Then 85s for the final SA
    
    best_overall_pts = None
    best_overall_score = 0.0
    
    # Phase 1: Quick exploration
    time_per_init = 5.0
    for i, init_pts in enumerate(inits):
        phase_deadline = start_time + (i + 1) * time_per_init
        if phase_deadline > deadline:
            break
        pts, score = _sa_run(init_pts, phase_deadline, seed=1000 + i)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
    
    # Phase 2: Long SA on the best found
    if best_overall_pts is not None and time.time() < deadline - 5.0:
        pts, score = _sa_run(best_overall_pts, deadline - 5.0, seed=999)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
    
    # Phase 3: Local polish
    if time.time() < deadline - 5.0 and best_overall_pts is not None:
        from scipy.optimize import minimize
        
        x0 = best_overall_pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 2)
            s, _ = _compute_score_and_worst(p)
            return -s
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_overall_score:
                best_overall_score = -res.fun
                best_overall_pts = res.x.reshape(N, 2)
        except Exception:
            pass
    
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = _initialize()
        
    return best_overall_pts

# EVOLVE-BLOCK-END
