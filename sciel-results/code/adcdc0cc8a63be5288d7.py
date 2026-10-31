"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin

    # The current best approach uses a hexagonal lattice subset.
    # Let's try to optimize a hexagonal lattice subset more carefully.
    # Or better yet, use a continuous optimization approach.
    
    # Objective: minimize (max_dist / min_dist)^2
    # Let points be P = [p_1, ..., p_16] in R^2
    # d_ij = ||p_i - p_j||
    # min_d = min_{i<j} d_ij
    # max_d = max_{i<j} d_ij
    # score = (max_d / min_d)^2
    
    # We can use a smooth approximation:
    # soft_max(d) = log(sum(exp(d^2 / T))) * T
    # soft_min(d) = -log(sum(exp(-d^2 / T))) * T
    # But this is tricky because of the ratio.
    
    # Alternative: minimize max_d^2 / min_d^2
    # Let's parameterize the points.
    # Due to symmetry, we can fix the first two points to remove translation and rotation invariance.
    # p_0 = (0, 0)
    # p_1 = (1, 0)  # fix scale to 1 for min_dist? No, that fixes min_dist potentially.
    # Actually, let's fix p_0 = (0,0) and p_1 = (1,0) and optimize the rest.
    # This removes 3 degrees of freedom (x-trans, y-trans, rotation).
    # Total variables: 14 * 2 = 28.
    
    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10  # Penalty for timeout
        # Reshape params to points
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Compute all pairwise distances
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_d = np.min(d)
        max_d = np.max(d)
        
        if min_d < 1e-8:
            return 1e10
        
        return (max_d / min_d)**2
    
    # Initial guess: hexagonal lattice points
    # Generate hex lattice points
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    # Sort by distance from origin
    hex_pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take 16 points
    initial_hex = np.array(hex_pts[:N])
    
    # Center and normalize
    initial_hex -= np.mean(initial_hex, axis=0)
    
    # Fix first two points?
    # Let's just run differential evolution on the full 32 parameters first, it's more robust.
    # Or use L-BFGS-B with the fixed points.
    
    # Let's try L-BFGS-B with fixed p0, p1
    # We need to transform the hex points to fit p0=(0,0), p1=(1,0)
    # This is complex. Let's just use differential_evolution on the 28 variables (pts[2:])
    
    # Bounds for differential evolution
    # Hex lattice points are roughly in [-5, 5]
    bounds = [(-5, 5)] * (14 * 2)
    
    # Use differential evolution
    # This might be slow. 28 dimensions.
    # Let's try a smaller population or use a better initial guess.
    
    # Let's try to optimize the hex lattice points directly.
    # Map the 16 hex points to the fixed frame.
    # Find the two closest points in the hex set.
    # Let's just use the first 16 points from the sorted list.
    
    # Actually, let's just run differential_evolution with a reasonable timeout.
    # But DE is slow.
    
    # Alternative: Use the hex lattice points as the initial guess for L-BFGS-B.
    # We need to set up the fixed points.
    # Let's pick the two points in the hex set that are closest to each other?
    # Or just the first two in the list?
    # The first two in the sorted list are (0,0) and (1,0) or (0, sqrt(3)/2)?
    # (0,0) is dist 0.
    # Next closest: (1,0), (0,1), (-1,0), etc. dist 1.
    # So pts[0] = (0,0), pts[1] = (1,0) is a good guess if they are in the set.
    
    # Let's check if (0,0) and (1,0) are in the top 16.
    # (0,0) is always first.
    # (1,0) is dist 1. There are 6 points at dist 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # So yes, (1,0) is likely in the top 16.
    
    # Let's extract the coordinates for pts[2:] from the hex set, 
    # transforming them so that pts[0]=(0,0) and pts[1]=(1,0).
    # This requires finding the index of (0,0) and (1,0) in the hex set.
    
    # Simpler: Just use the hex points as is, but optimize all 32 parameters with L-BFGS-B,
    # adding penalties for translation and rotation? No, that's messy.
    
    # Let's stick to the fixed frame approach.
    # Find indices of (0,0) and (1,0) in initial_hex.
    # (0,0) is at index 0 (sorted by dist).
    # (1,0) is one of the points at dist 1.
    
    # Let's just pick the first two points from the hex set as the anchor.
    # p0 = initial_hex[0]
    # p1 = initial_hex[1]
    
    # Compute transformation to map p0 -> (0,0), p1 -> (1,0)
    # Translation: -p0
    # Rotation: angle of (p1-p0)
    # Scale: 1 / ||p1-p0||
    
    p0 = initial_hex[0]
    p1 = initial_hex[1]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        # Degenerate case, fallback
        return initial_hex
    
    # Rotation matrix
    # We want to rotate v to (1, 0)
    # v = (vx, vy)
    # cos(theta) = vx/norm_v, sin(theta) = vy/norm_v
    # Rotation by -theta
    # R = [[cos, sin], [-sin, cos]]
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (initial_hex - p0) @ R_mat.T / norm_v
    
    # Now transformed[0] should be (0,0) and transformed[1] should be (1,0)
    # Initial params for L-BFGS-B are transformed[2:]
    init_params = transformed[2:].flatten()
    
    # Bounds for L-BFGS-B
    # Points should stay within a reasonable range
    bnds = [(-10, 10)] * 28
    
    # Run L-BFGS-B
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() > deadline:
            # Return the best found so far?
            # res.x is the current point
            pass
        
        best_params = res.x
        best_score = res.fun
        
    except Exception:
        # Fallback to hex lattice
        best_params = init_params
        best_score = objective(init_params)
    
    # Check if we have time for a second pass or a different method
    # Let's try differential_evolution if we have time
    if time.time() < deadline - 30:
        try:
            # Use a smaller population
            de_res = differential_evolution(objective, bounds, maxiter=50, 
                                            popsize=10, seed=42, 
                                            tol=1e-6, atol=1e-8)
            if de_res.fun < best_score:
                best_params = de_res.x
                best_score = de_res.fun
        except Exception:
            pass
    
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
