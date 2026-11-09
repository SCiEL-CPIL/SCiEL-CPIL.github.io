import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basin_hopping

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0

def _true_score(points: np.ndarray) -> float:
    """Compute the actual (d_min / d_max)^2 score."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return 0.0
    if d_max <= 1e-15:
        return 0.0
    if d_min <= 1e-15:
        return 0.0
    ratio = d_min / d_max
    return float(ratio * ratio)

def _soft_objective(points_flat: np.ndarray, beta: float = 20.0) -> float:
    """
    Smoothed objective function to minimize.
    Minimizes ln(soft_max) - ln(soft_min).
    This is equivalent to maximizing soft_min / soft_max.
    """
    points = points_flat.reshape(-1, 3)
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Avoid overflow in exp
    # soft_min = -1/beta * log(sum(exp(-beta * dists)))
    # soft_max =  1/beta * log(sum(exp( beta * dists)))
    
    # Numerical stability for log-sum-exp
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    # For soft_min:
    # log(sum(exp(-beta * d))) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
    # soft_min = min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
    
    # For soft_max:
    # log(sum(exp(beta * d))) = beta * max_d + log(sum(exp(beta * (d - max_d))))
    # soft_max = max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
    
    exp_neg = np.exp(-beta * (dists - min_d))
    soft_min = min_d - (1.0 / beta) * np.log(np.sum(exp_neg))
    
    exp_pos = np.exp(beta * (dists - max_d))
    soft_max = max_d + (1.0 / beta) * np.log(np.sum(exp_pos))
    
    if soft_min <= 0 or soft_max <= 0:
        return 1e10
        
    return np.log(soft_max) - np.log(soft_min)

def _local_optimize(points: np.ndarray, beta: float = 20.0) -> np.ndarray:
    """
    Perform local optimization using L-BFGS-B on the smoothed objective.
    """
    x0 = points.flatten()
    
    # Define the function for scipy
    def func(x):
        return _soft_objective(x, beta)
        
    # Try L-BFGS-B
    res = minimize(func, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        return res.x.reshape(-1, 3)
    else:
        # If L-BFGS-B fails, try Nelder-Mead
        res_nm = minimize(func, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res_nm.success:
            return res_nm.x.reshape(-1, 3)
        return points

def _generate_icosahedron() -> np.ndarray:
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    pts = np.array(verts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _add_random_points(pts: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Add random points on the sphere to pts."""
    new_pts = []
    for _ in range(count):
        # Random point on unit sphere
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        new_pts.append([x, y, z])
    new_pts = np.array(new_pts)
    return np.vstack([pts, new_pts])

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # Initial guesses
    ico = _generate_icosahedron()
    
    # Guess 1: Icosahedron + 2 random points on sphere
    rng = np.random.default_rng(42)
    guess1 = _add_random_points(ico, 2, rng)
    
    # Guess 2: 14 random points on sphere
    guess2 = _add_random_points(np.zeros((0,3)), 14, rng)
    
    # Guess 3: 14 points in a cube grid (scaled)
    # Just a few structured guesses
    grid_x = np.array([-1, 1, -0.5, 0.5, 0, -1, 1, -0.5, 0.5, 0, -1, 1, -0.5, 0.5])
    grid_y = np.array([-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1])
    grid_z = np.array([0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0])
    guess3 = np.column_stack((grid_x, grid_y, grid_z))
    # Normalize to sphere
    norms = np.linalg.norm(guess3, axis=1, keepdims=True)
    guess3 = guess3 / norms
    
    guesses = [guess1, guess2, guess3]
    
    # Iterate through guesses
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break
            
        # Run Basin Hopping
        # We need to define a step function for basin hopping
        def step(x, stepsize):
            return x + stepsize * rng.normal(size=x.shape)
            
        # Basin hopping minimizes a scalar. We use the soft objective.
        # Note: basin_hopping requires the objective to be a scalar function of x.
        
        def bh_obj(x):
            return _soft_objective(x, beta=20.0)
            
        try:
            # Run basin hopping
            # niter is the number of basin hopping iterations
            # We limit iterations based on time
            # Each local optimize takes ~0.1-0.5s? 
            # Let's allow ~10-20 basin hopping steps per guess if time permits.
            
            # Simple approach: Just run local optimize, then perturb, then local optimize.
            # Manual basin hopping for better time control.
            
            current_pts = guess.copy()
            current_score = _true_score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
            for bh_iter in range(20): # Max 20 hops per guess
                if time.time() > deadline:
                    break
                    
                # Local Optimize
                opt_pts = _local_optimize(current_pts, beta=20.0)
                opt_score = _true_score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
                    
                if opt_score > current_score:
                    current_pts = opt_pts
                    current_score = opt_score
                else:
                    # If local opt didn't improve, keep current for perturbation? 
                    # Or use the opt_pts as the new center?
                    # Standard basin hopping: accept if lower energy (higher score).
                    # Since we are maximizing, we accept if score improves.
                    # If it doesn't, we might still want to move.
                    # Let's just use the optimized points as the new center for perturbation.
                    current_pts = opt_pts
                    
                # Perturb
                # Step size should be moderate
                step_size = 0.5
                perturbed_pts = current_pts + step_size * rng.normal(size=current_pts.shape)
                
                # Optional: Project back to sphere? 
                # The problem doesn't require sphere constraint, but it helps.
                # Let's try without projection first, as the optimizer handles it.
                # However, scaling might drift. Let's normalize to unit sphere after perturbation?
                # If we normalize, we restrict the search to the sphere.
                # Given the high score of spherical codes, this is a good bet.
                norms = np.linalg.norm(perturbed_pts, axis=1, keepdims=True)
                perturbed_pts = perturbed_pts / norms
                
                current_pts = perturbed_pts
                
        except Exception:
            pass
            
    # Final fallback if no points found
    if best_points is None:
        best_points = guess1
        
    # Ensure shape and finiteness
    best_points = np.asarray(best_points, dtype=np.float64)
    if best_points.shape != (14, 3):
        best_points = guess1
        
    if not np.all(np.isfinite(best_points)):
        best_points = guess1
        
    return best_points
