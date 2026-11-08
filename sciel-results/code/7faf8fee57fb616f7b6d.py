"""
Strategy:
1. Model the problem as maximizing a smooth, differentiable lower bound on the minimum triangle area.
2. Use the log-sum-exp approximation: soft_min(a, t) = -t * log(sum(exp(-a_i/t)))
   As t -> 0, this approaches min(a_i).
3. Use L-BFGS-B from multiple random initializations.
4. Normalize the objective by the convex hull area to make the problem scale-invariant.
5. Use a time-budgeted loop to perform multiple restarts and polish the best solution.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14
DEADLINE = 148.0  # Safety margin before 150s kill

# Precompute triangle indices for N=14
def _generate_triangle_indices(n):
    inds = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                inds.append((i, j, k))
    return np.array(inds, dtype=int)

_TRI_INDICES = _generate_triangle_indices(N)
_M = len(_TRI_INDICES)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(pts):
    """
    Compute areas of all triangles formed by triples of points.
    """
    i, j, k = _TRI_INDICES.T
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hard_score(pts):
    """
    Compute min triangle area / hull area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
        
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
        
    return min_area / hull_a

def _objective(x, temp=1.0):
    """
    Objective function for L-BFGS-B.
    Minimizes -soft_min(areas) / hull_area.
    
    soft_min(areas, temp) = -temp * log(sum(exp(-areas/temp)))
    This is a smooth approximation of min(areas).
    
    We want to maximize min(areas) / hull_area.
    So we minimize - (soft_min(areas) / hull_area).
    
    To handle scale invariance, we can normalize points to unit hull area inside the objective,
    or just rely on the ratio. The ratio is scale invariant.
    However, L-BFGS-B might drift in scale if not constrained.
    Let's normalize the points to have unit hull area inside the objective calculation
    to keep the numbers stable and remove the scale degree of freedom.
    
    Steps:
    1. Reshape x to (N, 2).
    2. Check validity.
    3. Compute hull area. If invalid, return large penalty.
    4. Normalize points to unit hull area.
    5. Compute triangle areas of normalized points.
    6. Compute soft_min of these areas.
    7. Return -soft_min (since hull_area is now 1).
    """
    pts = x.reshape((N, 2))
    
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e6
        
    # Normalize to unit hull area
    scale = np.sqrt(hull_a)
    pts_norm = pts / scale
    
    areas = _triangle_areas(pts_norm)
    
    if len(areas) == 0:
        return 1e6
        
    # Soft min calculation
    # To avoid overflow in exp(-areas/temp), we can shift by the min area.
    # Let m = min(areas).
    # sum(exp(-areas/temp)) = exp(-m/temp) * sum(exp(-(areas-m)/temp))
    # log(sum(exp(-areas/temp))) = -m/temp + log(sum(exp(-(areas-m)/temp)))
    # soft_min = -temp * [ -m/temp + log(sum(exp(-(areas-m)/temp))) ]
    #           = m - temp * log(sum(exp(-(areas-m)/temp)))
    
    m = np.min(areas)
    shifted = areas - m
    # exp(-shifted/temp) is always <= 1, so no overflow.
    # If temp is very small, exp(-large) -> 0.
    exp_terms = np.exp(-shifted / temp)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Should not happen as shifted[0]=0 -> exp(0)=1
        return 0.0
        
    log_sum = np.log(sum_exp)
    soft_min_val = m - temp * log_sum
    
    # We want to maximize soft_min_val.
    # So we minimize -soft_min_val.
    return -soft_min_val

def _local_search(pts, time_limit=5.0):
    """
    Simple local search to polish the result.
    Uses small random perturbations and accepts improvements.
    """
    start_t = time.time()
    current_pts = pts.copy()
    current_score = _hard_score(current_pts)
    
    step = 0.01
    rng = np.random.default_rng(42)
    
    while time.time() - start_t < time_limit:
        improved = False
        # Try a few random directions
        for _ in range(50):
            direction = rng.normal(size=(N, 2))
            direction /= np.linalg.norm(direction)
            
            trial = current_pts + step * direction
            s = _hard_score(trial)
            if s > current_score:
                current_pts = trial
                current_score = s
                improved = True
            
            # Also try opposite direction? No, random is enough with many tries.
            
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
        else:
            # Keep step or increase slightly?
            # step *= 1.1 # Maybe too aggressive
            pass
            
    return current_pts, current_score

def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Try a few different temperatures for the log-sum-exp
    temperatures = [0.05, 0.1, 0.2, 0.5]
    
    # Number of restarts based on time
    # Each L-BFGS-B run takes ~0.5-1s.
    # We have ~140s.
    # Let's do 4 temps * 5 restarts = 20 runs.
    # Plus local search.
    
    restarts_per_temp = 5
    
    for temp in temperatures:
        for r in range(restarts_per_temp):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Generate initial point
            if r == 0:
                x_init = x0.copy()
            else:
                # Perturb regular polygon
                noise = np.random.normal(0, 0.1, size=(N, 2))
                x_init = (np.column_stack([np.cos(angles), np.sin(angles)]) + noise).flatten()
                
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective,
                    x_init,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if res.success:
                    pts = res.x.reshape((N, 2))
                    score = _hard_score(pts)
                    
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
                        
            except Exception as e:
                pass
                
        # Check time
        if time.time() - start_time > DEADLINE - 20:
            break
            
    if best_pts is None:
        # Fallback
        best_pts = x0.reshape((N, 2))
        best_score = _hard_score(best_pts)
        
    # Polish the best solution found
    if time.time() - start_time < DEADLINE - 5:
        polish_time = min(10.0, DEADLINE - 3 - (time.time() - start_time))
        if polish_time > 0:
            polished_pts, polished_score = _local_search(best_pts, time_limit=polish_time)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
                
    return best_pts

# EVOLVE-BLOCK-END
