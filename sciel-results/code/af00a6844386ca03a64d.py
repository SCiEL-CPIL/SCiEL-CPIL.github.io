"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _negative_score(params):
    """
    Objective function for optimization.
    params: flattened array of shape (2*N,) representing N points in 2D.
    """
    pts = params.reshape(-1, 2)
    
    # Prevent degenerate configurations (e.g., all points at origin)
    if np.allclose(pts, 0):
        return 1e9
        
    # Optional: Center the points to help optimization? 
    # Actually, Nelder-Mead handles translation well, but centering might reduce variance.
    # Let's stick to raw coordinates for simplicity, but ensure they don't drift to infinity.
    # The score is scale invariant, so the optimizer might try to scale points up/down.
    # To prevent this, we can normalize the RMS radius inside the objective or just let it find a local max.
    # However, unnormalized scale can cause numerical issues. 
    # Let's normalize the points to unit RMS radius before scoring to keep the search space compact.
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-6:
        return 1e9
    pts_norm = pts / rms
    
    return -_score(pts_norm)

def _get_initial_rings():
    """
    Generate the 2-ring initial configuration used by the parent program.
    Outer: 7 pts on unit circle. Inner: 7 pts on circle r=0.5, rotated by pi/7.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r_inner = 0.5
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Multi-start Nelder-Mead
    # We will try several starting points:
    # 1. The known good ring configuration
    # 2. A slightly perturbed version of the ring configuration
    # 3. A regular 14-gon (flat, likely bad, but good baseline)
    # 4. Random points in a unit disk
    
    starts = []
    
    # Start 1: Rings
    p_rings = _get_initial_rings()
    starts.append(p_rings.flatten())
    
    # Start 2: Perturbed Rings
    p_pert = p_rings + np.random.normal(0, 0.05, p_rings.shape)
    starts.append(p_pert.flatten())
    
    # Start 3: Regular 14-gon
    angles_14 = np.linspace(0, 2*np.pi, 14, endpoint=False)
    p_14 = np.column_stack([np.cos(angles_14), np.sin(angles_14)])
    starts.append(p_14.flatten())
    
    # Start 4: Random
    p_rand = np.random.uniform(-1, 1, (14, 2))
    # Ensure not collinear
    if np.linalg.matrix_rank(p_rand) < 3:
        p_rand[0] = [1, 0]
        p_rand[1] = [0, 1]
    starts.append(p_rand.flatten())

    for i, start_params in enumerate(starts):
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                _negative_score, 
                start_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True}
            )
            
            final_neg_score = res.fun
            final_score = -final_neg_score
            
            if final_score > best_score:
                best_score = final_score
                best_params = res.x.copy()
                
        except Exception:
            pass

    if best_params is None:
        # Fallback to rings if optimization failed
        pts = _get_initial_rings()
        return pts

    # Reconstruct points from best parameters
    pts = best_params.reshape(-1, 2)
    
    # Normalize to unit RMS radius for consistency
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms > 1e-6:
        pts = pts / rms
        
    return pts
# EVOLVE-BLOCK-END
