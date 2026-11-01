import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar, minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # Strategy:
    # Construct points from a regular octahedron (6 vertices) and a regular cube (8 vertices).
    # Octahedron vertices: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    # Cube vertices: (±1, ±1, ±1)
    
    # The relative size of the cube vs octahedron and the rotation of the cube
    # around the z-axis (or any axis, by symmetry z is fine) are the key parameters.
    # Let s be the scale factor for the cube vertices relative to the octahedron.
    # Let theta be the rotation angle of the cube around the z-axis.
    
    # Octahedron points (fixed)
    oct_pts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # Cube points (base, before rotation and scaling)
    cube_base = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    
    def generate_points(s, theta):
        """
        Generate 14 points given cube scale s and rotation angle theta.
        """
        # Rotate cube_base around z-axis by theta
        c = np.cos(theta)
        s_ = np.sin(theta)
        R = np.array([
            [c, -s_, 0],
            [s_, c, 0],
            [0, 0, 1]
        ])
        
        cube_pts = (cube_base @ R.T) * s
        all_pts = np.vstack([oct_pts, cube_pts])
        
        # Center the points to remove translational degree of freedom
        # The centroid of octahedron is (0,0,0).
        # The centroid of a symmetric cube is (0,0,0).
        # So the combined centroid is (0,0,0).
        # Thus, no translation needed if centered at origin.
        
        return all_pts
    
    # Objective function to minimize
    def objective(params):
        s, theta = params
        # Ensure s is positive
        if s < 1e-6:
            return 1e10
        pts = generate_points(s, theta)
        return _true_score(pts)
    
    # We will search over s in a reasonable range and theta in [0, pi/4] due to symmetry.
    # Symmetry of cube: pi/2 rotation maps cube to itself.
    # Symmetry of octahedron: pi/2 rotation maps oct to itself.
    # Combined symmetry might be smaller, but [0, pi/4] is a safe fundamental domain.
    
    best_score = np.inf
    best_params = None
    
    # Initial guess: 
    # If s=1, cube vertices are at distance sqrt(3) from origin, oct at 1.
    # Min dist might be between oct and cube.
    # Let's try s around 0.5 to 2.0
    
    # Use a grid search first to find a good starting point, then refine.
    s_vals = np.linspace(0.3, 2.0, 20)
    theta_vals = np.linspace(0, np.pi/4, 20)
    
    for s in s_vals:
        for theta in theta_vals:
            if time.time() - _t0 > DEADLINE - 5:
                break
            score = objective([s, theta])
            if score < best_score:
                best_score = score
                best_params = [s, theta]
    
    if best_params is None:
        # Fallback to a simple configuration
        return generate_points(1.0, 0.0)
    
    # Refine using local optimization
    # We can use Nelder-Mead or L-BFGS-B
    try:
        res = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a few other starting points in case we got stuck in a local minimum
    # The landscape might have multiple local minima due to symmetry breaking.
    start_points = [
        [0.5, 0.0],
        [0.8, np.pi/8],
        [1.2, np.pi/4],
        [1.5, 0.2],
        [0.7, np.pi/6],
        [1.0, np.pi/12]
    ]
    
    for sp in start_points:
        if time.time() - _t0 > DEADLINE - 5:
            break
        try:
            res = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    final_pts = generate_points(best_params[0], best_params[1])
    
    return final_pts
