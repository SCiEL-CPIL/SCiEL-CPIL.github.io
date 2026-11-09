# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective_function(flat_points):
    """
    Objective function for minimization: minimize d_max / d_min
    which is equivalent to maximizing d_min / d_max
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min == 0:
        return 1e10
    return d_max / d_min

def smooth_objective(flat_points, beta=100.0):
    """
    Smooth surrogate for the objective using log-sum-exp for max and soft-min for min.
    We want to maximize d_min / d_max, so minimize d_max / d_min.
    Let f = d_max / d_min.
    log(f) = log(d_max) - log(d_min).
    log(d_max) ~ logsumexp(log(dists))
    log(d_min) ~ -logsumexp(-log(dists))
    So we minimize exp(logsumexp(log(dists)) - (-logsumexp(-log(dists))))
    Actually, simpler: just minimize (soft_max(dists) / soft_min(dists)).
    soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
    soft_min(x, beta) = -(1/beta) * log(sum(exp(-beta*x)))
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    
    # Avoid log(0)
    dists = np.maximum(dists, 1e-10)
    
    log_dists = np.log(dists)
    
    # Stable log-sum-exp
    max_log = np.max(log_dists)
    soft_max = max_log + np.log(np.sum(np.exp(log_dists - max_log))) / beta
    
    min_log = np.min(log_dists)
    soft_min = min_log - np.log(np.sum(np.exp(-log_dists + min_log))) / beta
    
    # Minimize ratio
    return soft_max / soft_min

def generate_hex_lattice(n_points=16):
    """
    Generate a hexagonal lattice subset of n_points, centered.
    """
    # Generate a larger hex lattice and pick the n_points closest to center
    # Hex lattice basis vectors
    a = 1.0
    b = a * np.sqrt(3) / 2
    
    points_list = []
    # Generate enough points
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * a + j * a / 2
            y = j * b
            points_list.append([x, y])
    
    points = np.array(points_list)
    # Center
    points -= np.mean(points, axis=0)
    
    # Sort by distance from center
    dists_from_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_from_center)
    
    # Pick closest n_points
    selected = points[indices[:n_points]]
    
    # Normalize scale
    selected -= np.mean(selected, axis=0)
    scale = np.max(np.linalg.norm(selected, axis=1))
    if scale > 0:
        selected /= scale
    
    return selected

def generate_circular_ring(n_points=16, n_rings=2, points_per_ring=8):
    """
    Generate points on concentric rings.
    """
    points_list = []
    # Inner ring
    for i in range(points_per_ring):
        angle = 2 * np.pi * i / points_per_ring
        points_list.append([np.cos(angle) * 0.5, np.sin(angle) * 0.5])
    
    # Outer ring
    for i in range(points_per_ring):
        angle = 2 * np.pi * i / points_per_ring + np.pi / points_per_ring
        points_list.append([np.cos(angle) * 1.0, np.sin(angle) * 1.0])
    
    points = np.array(points_list[:n_points])
    points -= np.mean(points, axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points

def refine_points(points, time_limit):
    """
    Use L-BFGS-B to refine the points.
    """
    start_time = time.time()
    
    # Flatten points
    x0 = points.flatten()
    
    # Try smooth objective first, then exact
    best_x = x0
    best_val = objective_function(x0)
    
    # Multiple restarts with smooth objective
    for beta in [10.0, 50.0, 100.0, 200.0]:
        if time.time() - start_time > time_limit * 0.7:
            break
            
        try:
            res = minimize(smooth_objective, x0, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with exact objective
    if time.time() - start_time < time_limit * 0.9:
        try:
            res = minimize(objective_function, best_x, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)

def local_search(points, time_limit):
    """
    Perform local search with simulated annealing to maximize the score.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Simulated Annealing parameters
    temp = 0.005
    min_temp = 1e-8
    cooling_rate = 0.9995
    
    # Step size for perturbations
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Generate a random perturbation
        spread = np.std(current_points)
        if spread == 0:
            spread = 1.0
            
        # Perturbation magnitude
        delta = np.random.randn(2) * step_size * spread
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Simulated annealing acceptance
            if temp > min_temp:
                delta_score = new_score - current_score
                prob = np.exp(delta_score / temp)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        temp *= cooling_rate
        
        # Adaptive step size
        step_size *= 0.99995
        if step_size < 1e-6:
            step_size = 1e-6
    
    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    deadline = time.time() + 590
    
    # Initial guess: 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Run local search
    best_points = points_grid
    best_score = calculate_score(points_grid)
    
    # Try a few different initializations
    inits = []
    
    # 1. Standard Grid
    inits.append(points_grid)
    
    # 2. Hexagonal-like Grid
    inits.append(generate_hex_lattice(n))
    
    # 3. Circular rings
    inits.append(generate_circular_ring(n))
    
    # 4. Random
    np.random.seed(42)
    pts_rand = np.random.randn(n, d)
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # 5. Random with different seed
    np.random.seed(123)
    pts_rand2 = np.random.randn(n, d)
    pts_rand2 -= np.mean(pts_rand2, axis=0)
    inits.append(pts_rand2)
    
    # Allocate time for each init
    num_inits = len(inits)
    time_per_init = (time.time() + 590 - time.time()) / (num_inits + 2)  # +2 for refinement passes
    
    for init_pts in inits:
        if time.time() > deadline - 30:
            break
            
        # Run local search on this init
        end_time_for_this = time.time() + time_per_init
        res_points = local_search(init_pts, end_time_for_this - time.time())
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do refinement passes on the best result
    if time.time() < deadline - 20:
        remaining_time = deadline - 20 - time.time()
        # Split time between two refinement passes
        half_time = remaining_time / 2
        best_points = refine_points(best_points, half_time)
        best_score = calculate_score(best_points)
        
        if time.time() < deadline - 10:
            best_points = refine_points(best_points, deadline - 10 - time.time())
            best_score = calculate_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
