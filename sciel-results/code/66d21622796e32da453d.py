"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _compute_score(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                           (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _score_params(params):
    """Convert parameter vector to points and compute score (negated for minimization)."""
    n = len(params) // 2
    pts = np.array([[params[2*i], params[2*i+1]] for i in range(n)])
    return -_compute_score(pts)

def _make_symmetric_config(r_inner, angles_inner, r_outer, angles_outer):
    """Create a symmetric configuration with inner and outer rings."""
    pts = []
    for i, angle in enumerate(angles_inner):
        pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    for i, angle in enumerate(angles_outer):
        pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    return np.array(pts)

def construct():
    """Construct 13 points to maximize minimum triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Try various symmetric configurations
    # 13 = inner ring + outer ring
    # Possible splits: (1,12), (2,11), (3,10), (4,9), (5,8), (6,7), (7,6), etc.
    
    symmetric_configs = []
    
    # Try different splits
    for n_inner in range(1, 8):
        n_outer = 13 - n_inner
        if n_outer < 3:
            continue
        
        # Regular polygons
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        
        # Try different radius ratios
        for r_inner in [0.3, 0.5, 0.7]:
            for r_outer in [0.9, 1.0]:
                pts = _make_symmetric_config(r_inner, angles_inner, r_outer, angles_outer)
                symmetric_configs.append(pts)
        
        # Also try with offset angles
        for r_inner in [0.5]:
            for r_outer in [1.0]:
                for offset in [0, np.pi/n_outer]:
                    angles_outer_offset = angles_outer + offset
                    pts = _make_symmetric_config(r_inner, angles_inner, r_outer, angles_outer_offset)
                    symmetric_configs.append(pts)
    
    # Evaluate all symmetric configs
    for pts in symmetric_configs:
        if time.time() > deadline:
            break
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Strategy 2: Local optimization from best symmetric config
    if best_pts is not None and time.time() < deadline - 5:
        # Parameterize: use polar coordinates for symmetry preservation
        # Or just do direct optimization on the best config
        
        # Try local optimization
        x0 = best_pts.flatten()
        
        # Use Nelder-Mead for local optimization
        try:
            result = minimize(_score_params, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if time.time() < deadline:
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _compute_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts
        except Exception:
            pass
    
    # Strategy 3: Try a few random starts with local optimization
    if time.time() < deadline - 20:
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random start near unit circle
            angles = np.sort(np.random.uniform(0, 2*np.pi, 13))
            radii = np.random.uniform(0.8, 1.0, 13)
            pts = np.array([[r * np.cos(a), r * np.sin(a)] for r, a in zip(radii, angles)])
            
            x0 = pts.flatten()
            try:
                result = minimize(_score_params, x0, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
                
                if time.time() < deadline:
                    opt_pts = result.x.reshape(-1, 2)
                    opt_score = _compute_score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_pts = opt_pts
            except Exception:
                pass
    
    # Final check
    if best_pts is None:
        # Fallback to simple symmetric config
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_pts = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    return best_pts
# EVOLVE-BLOCK-END
