# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_points_symmetric(params):
    """
    Generate 14 points in 3D with D6h symmetry (2 rings of 6 + 2 poles).
    params: [r1, z1, r2, z2]
    - Ring 1: 6 points at radius r1, height z1, angles 0, 60, ..., 300
    - Ring 2: 6 points at radius r2, height z2, angles 30, 90, ..., 330 (twisted)
    - Poles: 2 points at (0,0, -z_p) and (0,0, z_p). 
      Note: To maintain 4 params, we fix z_p relative to z1/z2 or add it as a 5th?
      Let's use 5 params: [r1, z1, r2, z2, z_p]
      Actually, let's stick to 4 params for simplicity by fixing z_p = 1 (scale invariance allows us to fix one coordinate if we normalize later, but we need to be careful).
      
      Better: Fix the scale by normalizing the final points to the unit sphere.
      So we can use 4 parameters: [r1, z1, r2, z2]. The poles are at (0,0, -1) and (0,0, 1)? 
      No, that forces the poles to be at distance 2. The optimal solution might have poles closer or further.
      Let's use 5 parameters: [r1, z1, r2, z2, zp].
      Then normalize the whole set to unit sphere.
    """
    r1, z1, r2, z2, zp = params
    
    # Ensure radii are non-negative
    r1 = abs(r1)
    r2 = abs(r2)
    zp = abs(zp)
    
    # Ring 1 angles: 0, pi/3, ...
    angles1 = np.arange(6) * (np.pi / 3)
    # Ring 2 angles: pi/6, pi/2, ... (twisted by 30 degrees)
    angles2 = angles1 + np.pi / 6
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1), np.full(6, z1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2), np.full(6, z2)])
    poles = np.array([[0, 0, -zp], [0, 0, zp]])
    
    points = np.vstack([pts1, pts2, poles])
    
    # Normalize to unit sphere to handle scale invariance
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    points = points / norms[:, np.newaxis]
    
    return points

def _objective_symmetric(params):
    """
    Objective function for the symmetric parameterization.
    Returns negative score for minimization.
    """
    points = _generate_points_symmetric(params)
    s = _score(points)
    return -s

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # We will search over the 5 parameters: [r1, z1, r2, z2, zp]
    # Since we normalize, the absolute scale doesn't matter, only ratios.
    # Let's fix zp = 1.0 to reduce dimensionality to 4? 
    # If we fix zp=1, we lose the ability to vary the pole distance relative to rings.
    # However, due to scale invariance, we can fix one parameter. Let's fix zp = 1.0.
    # Then we optimize [r1, z1, r2, z2].
    
    # Grid search / Multi-start
    # Reasonable ranges for parameters relative to zp=1
    # r1, r2 in [0.1, 1.5]
    # z1, z2 in [-0.8, 0.8] (symmetry suggests z1 ~ -z2, but let's allow asymmetry)
    
    # To save time, we will use a coarse grid and then refine the top candidates.
    
    # Coarse Grid
    r_vals = np.linspace(0.2, 1.2, 15)
    z_vals = np.linspace(-0.7, 0.7, 15)
    
    candidates = []
    
    # Generate candidates from grid
    # We will subsample to keep the number of evaluations reasonable.
    # 15^4 = 50,625 evaluations. Each eval is fast (14 points). 
    # 50k * ~0.001s = 50s. This is acceptable.
    
    # To be safer, let's do a coarser grid first: 8^4 = 4096
    r_coarse = np.linspace(0.2, 1.2, 8)
    z_coarse = np.linspace(-0.7, 0.7, 8)
    
    scores = []
    params_list = []
    
    for r1 in r_coarse:
        for z1 in z_coarse:
            for r2 in r_coarse:
                for z2 in z_coarse:
                    # Fix zp = 1.0
                    params = np.array([r1, z1, r2, z2, 1.0])
                    try:
                        obj = _objective_symmetric(params)
                        score = -obj
                        if not np.isnan(score):
                            scores.append(score)
                            params_list.append(params)
                    except Exception:
                        pass
                    
                    if time.time() > deadline - 60:
                        break
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
        if time.time() > deadline - 60:
            break
            
    if scores:
        # Find top K candidates
        scores = np.array(scores)
        top_k = 10
        top_indices = np.argsort(scores)[-top_k:]
        
        # Refine top candidates using Nelder-Mead
        for idx in top_indices:
            if time.time() > deadline - 30:
                break
                
            initial_params = params_list[idx]
            
            try:
                res = minimize(
                    _objective_symmetric, 
                    initial_params, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 500}
                )
                
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = _generate_points_symmetric(res.x)
                    
            except Exception:
                continue
                
    # Fallback: If the symmetric search didn't find anything good, try a random non-symmetric search
    # This is unlikely to beat the symmetric one for N=14, but good for robustness.
    if best_params is None or best_score < 0.1:
        np.random.seed(42)
        # Try a few random starts with full 42D Nelder-Mead
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            pts = np.random.randn(14, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            
            def _obj_full(params):
                p = params.reshape(14, 3)
                p = p / np.linalg.norm(p, axis=1, keepdims=True)
                return -_score(p)
            
            try:
                res = minimize(
                    _obj_full, 
                    pts.flatten(), 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(14, 3)
                    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
            except Exception:
                pass

    # If still nothing, return a default good configuration (Icosahedron + 2 poles)
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        best_points = np.vstack([ico, extra])
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
