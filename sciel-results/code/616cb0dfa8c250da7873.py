"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci lattice method."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        radius_at_z = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        points[i, 0] = np.cos(theta) * radius_at_z
        points[i, 1] = np.sin(theta) * radius_at_z
        points[i, 2] = z
    return points

def construct():
    # 1. Initialize with Fibonacci lattice on a sphere
    # Scale to unit radius initially
    points = fibonacci_sphere(N)
    
    # 2. Simulated Annealing
    # Objective: minimize (max_dist / min_dist)^2
    # We work with log(ratio) or just ratio. Let's use ratio.
    
    def get_score(pts):
        dists = pdist(pts)
        if np.any(dists < 1e-10):
            return 1e10
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d == 0:
            return 1e10
        return (max_d / min_d) ** 2

    current_pts = points.copy()
    current_score = get_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995 # Cooling rate
    max_iter = 50000
    step_size = 0.1
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        # Generate new point
        # Perturb one or more points
        num_perturbed = 1
        idx = rng.choice(N, size=num_perturbed, replace=False)
        new_pts = current_pts.copy()
        new_pts[idx] += rng.normal(0, step_size, size=(num_perturbed, D))
        
        new_score = get_score(new_pts)
        
        # Metropolis criterion
        if new_score < current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta < 700 * T:
                prob = np.exp(-delta / T)
            else:
                prob = 0.0
            
            if rng.random() < prob:
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Adaptive step size (optional, but helps)
        # If we are accepting too few, increase step; if too many, decrease.
        # For simplicity, we keep it fixed or slightly decayed.
        step_size *= 0.9999
        
    # 3. Local Refinement with Nelder-Mead
    # Use the best point found by SA as the starting point for NM
    # NM can get stuck if the landscape is very rugged, but SA provides a good basin.
    
    def objective(x):
        pts = x.reshape(N, D)
        dists = pdist(pts)
        if np.any(dists < 1e-10):
            return 1e10
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d == 0:
            return 1e10
        return (max_d / min_d) ** 2

    x0 = best_pts.flatten()
    
    # Run NM for a limited number of iterations to polish
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    opt_points = res.x.reshape(N, D)
    
    # Verify the score
    final_score = get_score(opt_points)
    
    # If NM made it worse (unlikely but possible due to numerical issues), use best_pts
    if final_score > best_score:
        return best_pts
    else:
        return opt_points

# EVOLVE-BLOCK-END
