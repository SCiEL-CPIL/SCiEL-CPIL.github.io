# EVOLVE-BLOCK-START
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

def _objective_flat(params):
    """
    Objective function for minimization.
    params: 42-length array of coordinates for 14 points in 3D.
    Normalizes points to unit sphere to handle scale invariance.
    Returns negative score.
    """
    points = params.reshape(14, 3)
    
    # Normalize to unit sphere to remove scale invariance
    norms = np.linalg.norm(points, axis=1)
    norms[norms == 0] = 1.0
    points = points / norms[:, np.newaxis]
    
    s = _score(points)
    return -s

def _generate_icosahedron_seed():
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Add 2 points at poles
    extra = np.array([[0, 0, 1], [0, 0, -1]])
    pts = np.vstack([ico, extra])
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts.flatten()

def _generate_cube_seed():
    # 8 corners of a cube
    corners = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    # 6 face centers
    centers = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    pts = np.vstack([corners, centers])
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts.flatten()

def _generate_current_structure_seed():
    # Reconstruct the 8+4+2 structure from the current program to use as a seed
    # Using the best params found by the current program if possible, 
    # but since we don't have them, we use a generic good guess: a=1, b=1, c=1 (Cube)
    # or a=1, b=0.5, c=0.5
    a, b, c = 1.0, 0.7, 0.7
    
    pts_8 = np.array([
        [ a,  b,  c], [ a,  b, -c], [ a, -b,  c], [ a, -b, -c],
        [-a,  b,  c], [-a,  b, -c], [-a, -b,  c], [-a, -b, -c]
    ])
    pts_4 = np.array([
        [ b,  c,  a], [ b, -c,  a],
        [ b,  c, -a], [ b, -c, -a]
    ])
    pts_2 = np.array([
        [ c,  a,  b],
        [ c, -a,  b]
    ])
    points = np.vstack([pts_8, pts_4, pts_2])
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    return points.flatten()

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Generate a pool of initial guesses
    initial_guesses = []
    
    # 1. Symmetry-based seeds
    initial_guesses.append(_generate_icosahedron_seed())
    initial_guesses.append(_generate_cube_seed())
    initial_guesses.append(_generate_current_structure_seed())
    
    # 2. Perturbed symmetry seeds
    np.random.seed(42)
    for seed in [_generate_icosahedron_seed(), _generate_cube_seed(), _generate_current_structure_seed()]:
        for _ in range(5):
            noise = np.random.normal(0, 0.1, 42)
            initial_guesses.append(seed + noise)
            
    # 3. Random starts
    for _ in range(30):
        pts = np.random.normal(0, 1, (14, 3))
        norms = np.linalg.norm(pts, axis=1)
        norms[norms == 0] = 1
        pts = pts / norms[:, np.newaxis]
        initial_guesses.append(pts.flatten())
        
    # 4. A few more specific random structures
    for _ in range(10):
        # Random points on sphere
        pts = np.random.randn(14, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        initial_guesses.append(pts.flatten())

    # Optimization Loop
    for i, guess in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            # L-BFGS-B is efficient for high-dimensional smooth problems
            # We use a moderate maxiter to allow convergence but not hang
            res = minimize(
                _objective_flat, 
                guess, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
                
        except Exception:
            continue

    # If L-BFGS-B didn't find a good solution or failed, try Nelder-Mead on the best found
    # or a few random starts as a fallback, though L-BFGS-B should be robust.
    if best_params is None or best_score < 0:
         # Fallback to a known good configuration
         phi = (1 + np.sqrt(5)) / 2
         ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
         ])
         ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
         extra = np.array([[0, 0, 1], [0, 0, -1]])
         points = np.vstack([ico, extra])
         points = points / np.linalg.norm(points, axis=1, keepdims=True)
         return points

    # Final Polish: Run Nelder-Mead on the best parameters found by L-BFGS-B
    # Nelder-Mead can sometimes polish the local minimum better if the landscape is non-smooth
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                _objective_flat, 
                best_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000}
            )
            score_nm = -res_nm.fun
            if score_nm > best_score:
                best_score = score_nm
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Convert back to (14, 3) array
    points = best_params.reshape(14, 3)
    
    # Final normalization to ensure unit sphere (removes any drift)
    norms = np.linalg.norm(points, axis=1)
    norms[norms == 0] = 1
    points = points / norms[:, np.newaxis]
    
    return points

# EVOLVE-BLOCK-END
