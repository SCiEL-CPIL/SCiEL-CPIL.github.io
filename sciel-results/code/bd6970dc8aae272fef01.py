import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

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

def _spherical_to_cartesian(params):
    """
    Convert 28 spherical parameters (theta, phi for 14 points) to 42 Cartesian coordinates.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi]
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    
    points = np.column_stack((x, y, z))
    return points

def _repulsive_objective(params):
    """
    Objective function to minimize.
    Uses a repulsive potential model which is smoother than min/max ratio.
    We want to maximize the minimum distance, so we minimize the sum of inverse powers.
    """
    points = _spherical_to_cartesian(params)
    
    # Ensure points are valid
    if not np.all(np.isfinite(points)):
        return 1e10
        
    dists = pdist(points, metric='euclidean')
    
    # Handle coincident points
    if np.any(dists < 1e-9):
        return 1e10
        
    # Repulsive potential: sum of 1/d^k
    # Higher k makes it more sensitive to close pairs (like min distance)
    # Lower k makes it more sensitive to all pairs (like packing)
    k = 10
    repulsion = np.sum(1.0 / (dists ** k))
    
    return repulsion

def _refine_with_nelder_mead(initial_params, maxiter=2000):
    """
    Run Nelder-Mead on spherical coordinates.
    """
    # Bounds for theta: [0, pi], phi: [0, 2*pi]
    # Nelder-Mead doesn't handle bounds natively, but the periodicity and 
    # symmetry of the problem mean that local minima are usually well-behaved 
    # within the fundamental domain. We can wrap angles if needed, but 
    # typically NM stays close to the initial guess if it's good.
    
    try:
        res = minimize(
            _repulsive_objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        return res.x
    except Exception:
        return initial_params

def _get_initial_guesses():
    """
    Generate diverse initial guesses for the 28D spherical space.
    """
    guesses = []
    
    # 1. Icosahedron-like seed
    # 12 vertices of icosahedron + 2 poles
    phi_golden = (1 + np.sqrt(5)) / 2
    ico_xyz = np.array([
        [0, 1, phi_golden], [0, -1, phi_golden], [0, 1, -phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1]
    ])
    ico_xyz = ico_xyz / np.linalg.norm(ico_xyz, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    seed_pts = np.vstack([ico_xyz, poles])
    
    # Convert to spherical
    thetas = np.arccos(seed_pts[:, 2])
    phis = np.arctan2(seed_pts[:, 1], seed_pts[:, 0])
    phis[phis < 0] += 2 * np.pi
    
    params = np.zeros(28)
    params[0::2] = thetas
    params[1::2] = phis
    guesses.append(params)
    
    # 2. Cube-like seed (8 corners + 6 face centers? No, 14 points. 
    #    Maybe 8 corners of cube + 6 centers of faces? That's 14. 
    #    But cube corners are closer to each other than to face centers? 
    #    Let's just use random perturbations of the icosahedron seed)
    
    np.random.seed(42)
    for i in range(20):
        # Perturb the icosahedron seed
        noise_theta = np.random.normal(0, 0.2, 14)
        noise_phi = np.random.normal(0, 0.2, 14)
        
        p = params.copy()
        p[0::2] += noise_theta
        p[1::2] += noise_phi
        
        # Wrap angles
        p[0::2] = np.clip(p[0::2], 0, np.pi)
        p[1::2] = p[1::2] % (2 * np.pi)
        
        guesses.append(p)
        
    # 3. Random uniform on sphere
    for i in range(10):
        # Random directions
        u = np.random.random(14)
        v = np.random.random(14)
        
        thetas = np.arccos(1 - 2 * u)
        phis = 2 * np.pi * v
        
        p = np.zeros(28)
        p[0::2] = thetas
        p[1::2] = phis
        guesses.append(p)
        
    return guesses

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -np.inf
    best_points = None
    best_params = None
    
    guesses = _get_initial_guesses()
    
    # Multi-start optimization
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        # 1. Coarse search with Nelder-Mead
        refined_params = _refine_with_nelder_mead(guess, maxiter=1000)
        
        # Convert to Cartesian to score
        pts = _spherical_to_cartesian(refined_params)
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            continue
            
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_params = refined_params.copy()
            
        # 2. Fine polish with L-BFGS-B on Cartesian coords? 
        # Or continue NM with tighter tolerances?
        # Let's do a quick L-BFGS-B on Cartesian starting from this point.
        # This helps escape small NM local minima.
        
        if time.time() < deadline - 10:
            try:
                def _obj_cart(params):
                    p = params.reshape(14, 3)
                    # Normalize to sphere
                    norms = np.linalg.norm(p, axis=1)
                    norms[norms == 0] = 1.0
                    p = p / norms[:, np.newaxis]
                    # Use repulsive potential again for smoothness
                    dists = pdist(p, metric='euclidean')
                    if np.any(dists < 1e-9):
                        return 1e10
                    return np.sum(1.0 / (dists ** 10))
                
                res = minimize(
                    _obj_cart,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                polished_pts = res.x.reshape(14, 3)
                norms = np.linalg.norm(polished_pts, axis=1)
                norms[norms == 0] = 1.0
                polished_pts = polished_pts / norms[:, np.newaxis]
                
                polished_score = _score(polished_pts)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts
            except Exception:
                pass

    # Fallback
    if best_points is None:
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_xyz = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden], [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_xyz = ico_xyz / np.linalg.norm(ico_xyz, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1], [0, 0, -1]])
        best_points = np.vstack([ico_xyz, poles])
        
    # Final safety check
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.0, neginf=0.0)
    if best_points.shape != (14, 3):
        best_points = np.eye(14, 3)
        
    return best_points

# EVOLVE-BLOCK-END
