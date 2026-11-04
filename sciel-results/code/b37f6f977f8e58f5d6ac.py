"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
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
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
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

def _negative_score(pts_flat):
    """
    Objective function for minimization.
    Reshapes flat array to (N, 2), computes score, returns negative.
    """
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin before 150s kill

    best_score = -float('inf')
    best_pts = None
    
    # Initial guess: Regular 14-gon on unit circle
    # This is a strong symmetric baseline.
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_flat = initial_pts.flatten()
    
    # Evaluate initial
    current_score = _score(initial_pts)
    best_score = current_score
    best_pts = initial_pts.copy()
    
    # Strategy:
    # 1. Run Powell's method (derivative-free, handles non-smoothness better than NM sometimes)
    #    Powell's method is often more robust for max-min problems than Nelder-Mead in high dimensions
    #    because it doesn't rely on simplex reflection which can get stuck in valleys.
    # 2. If time permits, try a few random restarts or perturbations.
    
    # Powell's method can be slow in high dimensions (28D).
    # We limit iterations and use maxiter.
    
    try:
        res = minimize(
            _negative_score, 
            initial_flat, 
            method='Powell', 
            options={
                'maxiter': 200, 
                'maxfev': 2000, 
                'xatol': 1e-8, 
                'fatol': 1e-8,
                'disp': False
            }
        )
        
        if time.time() < deadline:
            candidate_pts = res.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts
    except Exception:
        pass
    
    # If time remains, try Nelder-Mead from the best point found by Powell
    # Powell often finds a good basin, NM can polish it if the landscape is smoother there.
    if time.time() < deadline - 5.0 and best_pts is not None:
        try:
            res_nm = minimize(
                _negative_score, 
                best_pts.flatten(), 
                method='Nelder-Mead', 
                options={
                    'maxiter': 500, 
                    'xatol': 1e-9, 
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            candidate_pts = res_nm.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts
        except Exception:
            pass
            
    # If time remains, try a different initial guess: 
    # Two rings (7 outer, 7 inner) but optimized in full space.
    # This breaks the strict symmetry assumption of the previous approach.
    if time.time() < deadline - 10.0:
        # Generate a two-ring initial guess
        r_inner = 0.6
        theta_offset = np.pi / 7
        angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
        
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        inner_pts = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        
        two_ring_pts = np.vstack([outer_pts, inner_pts])
        two_ring_flat = two_ring_pts.flatten()
        
        # Quick Powell on this specific topology start
        try:
            res_powell_2 = minimize(
                _negative_score, 
                two_ring_flat, 
                method='Powell', 
                options={
                    'maxiter': 100, 
                    'maxfev': 1000,
                    'xatol': 1e-8, 
                    'fatol': 1e-8
                }
            )
            candidate_pts = res_powell_2.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts
                
                # Polish with NM if time allows
                if time.time() < deadline - 2.0:
                    res_nm_2 = minimize(
                        _negative_score, 
                        best_pts.flatten(), 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 200, 
                            'xatol': 1e-9, 
                            'fatol': 1e-9,
                            'adaptive': True
                        }
                    )
                    candidate_pts = res_nm_2.x.reshape(N, 2)
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
