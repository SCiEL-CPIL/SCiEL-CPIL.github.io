import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 100

def construct():
    # Fix first point at North Pole to break rotational symmetry
    # Optimize the remaining N-1 points using spherical coordinates
    # theta in [0, pi], phi in [0, 2*pi]
    
    n_opt = N - 1
    # Parameters: theta_1..theta_n, phi_1..phi_n
    # Total dims: 2 * (N-1)
    
    def get_points(params):
        # params: shape (2 * n_opt,)
        thetas = params[:n_opt]
        phis = params[n_opt:]
        
        # Generate optimized points
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        
        opt_points = np.column_stack([x, y, z])
        
        # Prepend fixed point
        fixed_point = np.array([[0.0, 0.0, 1.0]])
        
        return np.vstack([fixed_point, opt_points])

    def energy(params):
        points = get_points(params)
        
        # Calculate pairwise distances
        # Use broadcasting to get all pairs
        # dist[i, j] = ||points[i] - points[j]||
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # We need sum over i < j of log(1 / dist)
        # Create upper triangular indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Check for zero distances (invalid)
        if np.any(pair_dists < 1e-12):
            return 1e10 # Large penalty
        
        # Energy = sum(log(1/d)) = -sum(log(d))
        # Using math.fsum for precision as requested in task description?
        # "pair terms summed with math.fsum"
        # Converting numpy array to list for math.fsum might be slow but N=100 is small (4950 terms)
        # Let's try to be efficient first, but respect the contract.
        # The contract says "Score: ... summed with math.fsum".
        # This implies the EVALUATOR uses math.fsum.
        # My internal optimization objective just needs to be consistent.
        # I will use numpy sum for speed in the optimizer, as it's monotonic with fsum.
        
        # log(1/d) = -log(d)
        return -np.sum(np.log(pair_dists))

    # Bounds for DE
    bounds = []
    for _ in range(n_opt):
        bounds.append((0.0, np.pi)) # theta
    for _ in range(n_opt):
        bounds.append((0.0, 2.0 * np.pi)) # phi
        
    # Initial population
    # Use Fibonacci spiral for the first few individuals to provide a good start
    def fib_spiral_params():
        # Generate N points
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # We fixed point 0 at (0,0,1).
        # We need to rotate the fib spiral such that one point is at (0,0,1)?
        # Or just use the fib spiral as a random-ish start for the 99 optimized points?
        # Actually, DE doesn't strictly need a perfect start, but it helps.
        # Let's just return None and let DE initialize randomly, or provide a custom init.
        # Providing a custom init with the fib spiral (ignoring the fixed point constraint for the init generation, 
        # then projecting? No, the parameters must be valid).
        # Let's just use the fib spiral points 1..99 as the initial params for one individual.
        # Point 0 of fib spiral is at z = 1 - 1/100 = 0.99. Not exactly (0,0,1).
        # We can just use the fib spiral coordinates for the 99 optimized points directly.
        # x,y,z -> theta, phi
        opt_pts = pts[1:] # Ignore the first one, use the rest as initial guess for optimized ones
        thetas = np.arccos(opt_pts[:, 2])
        phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
        
        # Ensure phis are in [0, 2pi]
        phis = phis % (2 * np.pi)
        
        return np.concatenate([thetas, phis])
        
    init_params = fib_spiral_params()
    
    # Differential Evolution
    # maxiter: 120s limit. 
    # Let's estimate. 
    # D=198. Pop size = 10*D = 1980.
    # Each iter: 1980 evals.
    # Eval time: ~1ms? (100 points, vectorized).
    # 1980 ms per iter ~ 2s.
    # 50 iters ~ 100s.
    # Let's try maxiter=50, popsize=10.
    
    try:
        res = differential_evolution(
            energy, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6,
            # init='sobol' or 'latinhypercube' is better than random
            init='latinhypercube',
            # x0=init_params # DE doesn't take x0 directly in this way for all versions, 
                           # but we can use 'init' parameter if available or just rely on random.
                           # Actually, scipy DE allows 'init' to be a string.
                           # To use custom init, we might need to patch or just run it.
                           # Let's stick to standard DE.
        )
        best_params = res.x
    except Exception:
        # Fallback to Fibonacci if DE fails
        best_params = init_params

    # Polish with local optimization
    # L-BFGS-B is good for box constraints
    try:
        res_local = minimize(
            energy, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        if res_local.success:
            best_params = res_local.x
    except Exception:
        pass

    final_points = get_points(best_params)
    
    # Normalize to ensure unit sphere (numerical stability)
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    # Check for invalid points (distance < 1e-9)
    # If invalid, return Fibonacci spiral as fallback
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    if np.any(pair_dists < 1e-9) or not np.all(np.isfinite(final_points)):
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return final_points
