"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

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

def _normalize_radius(pts, target_rms=1.0):
    """
    Normalize the point set so that the RMS distance from the centroid is target_rms.
    This keeps the scale fixed, which is important because the score is scale-invariant.
    """
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if rms < 1e-12:
        return pts
    scale = target_rms / rms
    return centered * scale + centroid

def _initial_construction_regular():
    """14 points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _initial_construction_rings():
    """2 rings: 7 outer, 7 inner."""
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r_outer = 1.0
    r_inner = 0.5
    
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # List of initial configurations
    inits = [
        _initial_construction_regular(),
        _initial_construction_rings()
    ]
    
    best_score = -float('inf')
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    T0 = 0.05  # Initial temperature
    T_end = 1e-6
    alpha = 0.995 # Cooling factor
    step_size = 0.1
    
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        current_pts = init_pts.copy()
        current_pts = _normalize_radius(current_pts)
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
        T = T0
        steps = 0
        
        while time.time() < deadline and T > T_end:
            steps += 1
            # Generate perturbation
            # Move a random point by a random vector
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_size, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] += delta
            
            # Normalize to maintain scale
            new_pts = _normalize_radius(new_pts)
            
            new_score = _score(new_pts)
            
            # Acceptance probability
            if new_score > current_score:
                accept = True
            else:
                delta_E = current_score - new_score
                if delta_E < 1e-12:
                    accept = False
                else:
                    accept = rng.random() < np.exp(-delta_E / T)
            
            if accept:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # If rejected, we might want to reduce step size slightly or just cool down
                pass
                
            T *= alpha
            
            # If we've done many steps, we can break early if T is very low
            if T < T_end:
                break

    if best_pts is None:
        # Fallback
        best_pts = inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
