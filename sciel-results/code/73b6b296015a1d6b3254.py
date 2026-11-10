import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    # FCC Lattice Generation
    # FCC basis vectors
    # We want nearest neighbor distance ~ 0.22
    # d_nn = a / sqrt(2) => a = d_nn * sqrt(2)
    # Let's try a range of 'a' or just one good one.
    # The optimal radius is ~0.1114, so d_nn ~ 0.2228.
    # a ~ 0.315
    
    a = 0.315
    a1 = np.array([0, a/np.sqrt(2), a/np.sqrt(2)]) # Wait, standard FCC vectors
    # Standard FCC primitive vectors:
    # v1 = a/2 * (0, 1, 1)
    # v2 = a/2 * (1, 0, 1)
    # v3 = a/2 * (1, 1, 0)
    # Nearest neighbor distance is a/sqrt(2).
    
    v1 = np.array([0, 0.5, 0.5]) * a
    v2 = np.array([0.5, 0, 0.5]) * a
    v3 = np.array([0.5, 0.5, 0]) * a
    
    # Generate lattice points
    # Range: cube is [-0.5, 0.5]. 
    # Max coordinate is 0.5.
    # i, j, k should cover enough range.
    # Let's use -4 to 4.
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                    points.append(p)
    
    points = np.array(points)
    
    # If we have more than N points, we need to select N.
    # If we have fewer, we might need to adjust 'a' or add points.
    # With a=0.315, volume per point ~ a^3 / 4? No, 4 points per cell of volume a^3.
    # Density ~ 4 / a^3.
    # 4 / 0.315^3 ~ 4 / 0.031 ~ 129 points per unit volume.
    # So we should have ~129 points. This is close to 100.
    
    if len(points) < N:
        # Reduce 'a' to get more points
        a = 0.28
        v1 = np.array([0, 0.5, 0.5]) * a
        v2 = np.array([0.5, 0, 0.5]) * a
        v3 = np.array([0.5, 0.5, 0]) * a
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                for k in range(-5, 6):
                    p = i * v1 + j * v2 + k * v3
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
        points = np.array(points)

    best_radius = 0
    best_centers = None
    rng = np.random.default_rng(42)
    
    # Try multiple random subsets and optimizations
    n_tries = 10
    
    for _ in range(n_tries):
        if len(points) > N:
            idx = rng.choice(len(points), N, replace=False)
            centers = points[idx].copy()
        else:
            centers = points.copy()
            # If still not enough, pad with random points? 
            # But for this specific a, we should have enough.
            if len(centers) < N:
                # Fallback: simple grid
                m = int(np.ceil(N ** (1.0 / 3.0)))
                ticks = (np.arange(m) + 0.5) / m - 0.5
                grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
                centers = grid[:N]
                break

        # Optimization
        # Objective: minimize sum of 1/d^p
        p_power = 10
        eps = 1e-9
        
        def objective(x):
            c = x.reshape(N, 3)
            # Pairwise distances
            # Compute all pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Mask diagonal
            np.fill_diagonal(dists, 1.0)
            # Clip to avoid division by zero
            dists = np.maximum(dists, eps)
            pair_term = np.sum(1.0 / np.power(dists, p_power))
            
            # Boundary distances
            # Distance to each face
            dx = 0.5 - np.abs(c[:, 0])
            dy = 0.5 - np.abs(c[:, 1])
            dz = 0.5 - np.abs(c[:, 2])
            b_dists = np.minimum(np.minimum(dx, dy), dz)
            b_dists = np.maximum(b_dists, eps)
            bound_term = np.sum(1.0 / np.power(b_dists, p_power))
            
            return pair_term + bound_term
        
        def gradient(x):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            dists = np.maximum(dists, eps)
            
            # Gradient of sum 1/d^p wrt c_i
            # d/dx_i (1/d_ij^p) = -p * d_ij^(-p-1) * (c_i - c_j) / d_ij = -p * (c_i - c_j) / d_ij^(p+2)
            # Sum over j != i
            # grad_i = -p * sum_{j!=i} (c_i - c_j) / ||c_i - c_j||^(p+2)
            
            # Vectorized calculation
            # dists shape (N, N)
            # We need to sum (c_i - c_j) / dists[i,j]^(p+2)
            
            # Create a weight matrix W[i,j] = 1 / dists[i,j]^(p+2)
            # Set diagonal to 0
            W = 1.0 / np.power(dists, p + 2)
            np.fill_diagonal(W, 0.0)
            
            # grad_pair = -p * (c_i * sum(W[i,:]) - sum(W[i,:] * c_j))
            # sum(W[i,:]) is row sum
            row_sums = np.sum(W, axis=1)
            # sum(W[i,:] * c_j) is W @ c
            weighted_c = W @ c
            
            grad_pair = -p * (c * row_sums[:, np.newaxis] - weighted_c)
            
            # Boundary gradient
            # d/dx_i (1/B_i^p)
            # B_i = min(0.5 - |x_i|, ...)
            # If x_i > 0, B_i = 0.5 - x_i (if this is the min). 
            # This is non-differentiable at 0 and at the min switch points.
            # Approximation: use sub-gradient or just numerical gradient?
            # L-BFGS-B can handle non-smoothness somewhat, but analytical is better.
            # Let's compute numerical gradient for the boundary part or just use finite differences for the whole thing?
            # Finite difference for 300 variables is expensive.
            # Let's stick to analytical for pairs and approximate for boundaries.
            # Actually, the boundary term is only active near the boundary.
            # For simplicity and robustness, let's use `method='L-BFGS-B'` without providing gradient, 
            # letting SciPy use finite differences. With 300 vars, it's 301 function evals per step.
            # If it takes 50 steps, that's 15000 evals. 
            # Each eval is O(N^2) = 10000 ops. 150M ops. This might be too slow in Python/NumPy if not vectorized well.
            # My objective function IS vectorized. 
            # 15000 * 10000 = 150,000,000 operations. 
            # NumPy can do ~100M ops/sec. So ~1.5 seconds per optimization.
            # 10 tries = 15 seconds. This is acceptable.
            
            return None # Use finite differences
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(objective, centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success or res.fun < 1e6:
                final_centers = res.x.reshape(N, 3)
                
                # Calculate actual radius
                diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, 1.0)
                min_dist = np.min(dists)
                
                dx = 0.5 - np.abs(final_centers[:, 0])
                dy = 0.5 - np.abs(final_centers[:, 1])
                dz = 0.5 - np.abs(final_centers[:, 2])
                min_bound = np.min(np.minimum(np.minimum(dx, dy), dz))
                
                r = 0.5 * min(min_dist, min_bound)
                
                if r > best_radius:
                    best_radius = r
                    best_centers = final_centers
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
