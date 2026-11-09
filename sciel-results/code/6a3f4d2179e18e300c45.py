import numpy as np
import scipy.optimize as opt
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Initialization: Hexagonal Lattice
    # We want to place N points in a unit circle.
    # Approximate radius of hex lattice packing:
    # Area per circle ~ sqrt(3)/2 * s^2. Total area ~ pi.
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate hex grid points
    k_max = int(1.0 / s) + 2
    points = []
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    # Filter points inside the unit circle
    dists = np.linalg.norm(points, axis=1)
    points = points[dists <= 1.0]
    
    # If we have more than N points, select the ones closest to the center? 
    # Or just take the first N? 
    # Better: Sort by distance to center and take the innermost N to ensure a compact start.
    # Actually, for packing, we want to fill the space. 
    # The hex lattice cut to a circle is a good start.
    # If len(points) < N, we might need to add points or adjust s.
    # If len(points) > N, we can randomly subsample or keep the ones that form a good distribution.
    # Let's just take the first N after sorting by distance to center to ensure they are inside.
    # But wait, if we cut the hex grid, we get a hexagonal shape. 
    # To get a circular shape, we just filter by radius.
    
    if len(points) < N:
        # If not enough points, reduce spacing
        s *= 0.9
        while len(points) < N:
            k_max = int(1.0 / s) + 2
            points = []
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3) / 2.0
                    points.append([x, y])
            points = np.array(points)
            dists = np.linalg.norm(points, axis=1)
            points = points[dists <= 1.0]
            s *= 0.9
            if time.time() > deadline:
                break
        
        if len(points) < N:
            # Pad with random points if still not enough (unlikely)
            pad = np.random.rand(N - len(points), 2) * 0.5
            points = np.vstack([points, pad])
        else:
            # Select N points closest to center to ensure they are well inside
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            points = points[indices]
    else:
        # We have more than N points. 
        # To maintain the lattice structure, we can just take the first N? 
        # Or better, randomly select N to break symmetry?
        # Let's take the N points closest to the center.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        points = points[indices]

    # Ensure finite
    points = np.nan_to_num(points)
    
    def objective_and_grad(x, lam):
        # x is shape (N, 2)
        # Calculate pairwise distances
        # d[i, j] = ||x[i] - x[j]||
        # To save memory and time, we can compute gradients analytically.
        
        # Vectorized distance calculation is O(N^2).
        # For N=300, 90000 distances. This is fast.
        
        # Compute all pairwise distances
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This creates a 300x300x2 array. ~ 1.8M floats. OK.
        
        # However, for gradient, we need to sum over pairs.
        
        # Let's use a loop over i for gradient accumulation to save memory?
        # No, vectorization is faster.
        
        # Calculate S = sum_{i<j} exp(-lam * d_ij) + sum_i exp(-lam * (1 - |c_i|))
        
        # 1. Boundary terms
        r = np.linalg.norm(x, axis=1)
        # Handle r=0
        r_safe = np.where(r > 1e-8, r, 1e-8)
        boundary_vals = 1.0 - r
        # Clip to avoid negative exponent issues? No, exp handles it.
        # But if 1-r is negative (outside), exp(-lam * negative) = exp(lam * positive) -> large.
        # This is good, it penalizes being outside.
        
        # 2. Pairwise terms
        # To avoid N^2 memory if N was larger, but N=300 is fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self
        np.fill_diagonal(dists, np.inf)
        
        # We only want i < j. 
        # The sum in log-sum-exp is over all pairs. 
        # If we sum over all i!=j, we count each pair twice.
        # This doesn't change the argmin, just the scale of the gradient.
        # Let's just sum over all i!=j for simplicity.
        
        # Log-sum-exp trick for stability
        # We want log(sum(exp(-lam * v)))
        # Let v_min = min(v)
        # log(sum(exp(-lam * v))) = -lam * v_min + log(sum(exp(-lam * (v - v_min))))
        
        all_vals = np.concatenate([boundary_vals, dists.flatten()])
        # Filter out inf
        all_vals = all_vals[~np.isinf(all_vals)]
        
        if len(all_vals) == 0:
            return 0.0, np.zeros_like(x)
            
        v_min = np.min(all_vals)
        
        # Calculate weights
        # We need to be careful with the gradient.
        # Let's compute the gradient directly.
        
        # Grad of boundary term for point i:
        # d/dx_i [ -1/lam * log(sum exp(-lam v_k)) ]
        # = -1/lam * 1/S * sum_k exp(-lam v_k) * (-lam) * d v_k / d x_i
        # = 1/S * sum_k exp(-lam v_k) * d v_k / d x_i
        
        # Let's compute S = sum exp(-lam v_k)
        # And sum_k exp(-lam v_k) * grad v_k
        
        # To avoid overflow/underflow, we can compute:
        # S = exp(-lam v_min) * sum exp(-lam (v_k - v_min))
        # Let T = sum exp(-lam (v_k - v_min))
        # S = exp(-lam v_min) * T
        
        # Term = 1/S * sum exp(-lam v_k) * grad v_k
        # = 1/(exp(-lam v_min) T) * exp(-lam v_min) * sum exp(-lam (v_k - v_min)) * grad v_k
        # = 1/T * sum exp(-lam (v_k - v_min)) * grad v_k
        
        # So we just need T and the weighted sum of gradients.
        
        # 1. Boundary Gradients
        # v_i = 1 - r_i
        # grad v_i = -x_i / r_i
        # weight_i = exp(-lam (v_i - v_min))
        
        weights_boundary = np.exp(-lam * (boundary_vals - v_min))
        # grad_boundary is shape (N, 2)
        grad_boundary = -x / r_safe[:, np.newaxis]
        # Where r is 0, grad is 0? The force is undefined, but we can set it to 0.
        grad_boundary[r < 1e-8] = 0
        
        sum_grad_boundary = np.sum(weights_boundary[:, np.newaxis] * grad_boundary, axis=0)
        
        # 2. Pairwise Gradients
        # v_ij = d_ij
        # grad v_ij w.r.t x_i = (x_i - x_j) / d_ij
        # grad v_ij w.r.t x_j = (x_j - x_i) / d_ij
        
        # weights_pair = exp(-lam (d_ij - v_min))
        # We need to accumulate for each i.
        
        # dists is (N, N)
        # weights_pair = exp(-lam (dists - v_min))
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(weights_pair := np.exp(-lam * (dists - v_min)), 0.0)
        
        # grad_dists[i, j] = (x_i - x_j) / d_ij
        # We can compute this as diff / dists[..., np.newaxis]
        # But dists has 0 on diagonal, causing div by 0.
        # We already set weights to 0 on diagonal, so we can mask the division.
        
        # To avoid division by zero, use np.where
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        grad_dists = diff / dists_safe[..., np.newaxis]
        
        # grad_dists[i, j] is gradient w.r.t x_i of distance d_ij.
        # The contribution to grad x_i is sum_j weights_pair[i, j] * grad_dists[i, j]
        
        sum_grad_pair = np.sum(weights_pair[..., np.newaxis] * grad_dists, axis=1)
        
        # Total Gradient
        grad = sum_grad_boundary + sum_grad_pair
        
        # Objective Value
        # f = -1/lam * log S = -1/lam * (-lam v_min + log T) = v_min - 1/lam log T
        # T = sum_boundary + sum_pair
        T = np.sum(weights_boundary) + np.sum(weights_pair)
        
        if T == 0:
            return 0.0, grad
            
        obj = v_min - (1.0 / lam) * np.log(T)
        
        return obj, grad

    # Optimization Stages
    # Start with low lambda (smooth), increase to high lambda (sharp)
    lambdas = [10.0, 50.0, 200.0, 1000.0]
    
    x = points.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        # Use BFGS
        # maxiter should be reasonable
        # We don't have a fixed budget per stage, but we check time.
        
        # Define a wrapper to check time
        result = opt.minimize(
            objective_and_grad, 
            x, 
            args=(lam,),
            jac=True,
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-5}
        )
        
        if result.success or result.fun < objective_and_grad(x, lam)[0]:
            x = result.x
        
        # Check time
        if time.time() > deadline - 10:
            break

    # Final check and return
    # Ensure points are inside and finite
    x = np.nan_to_num(x)
    
    # One last polish? Maybe a few more steps with high lambda?
    # If time permits.
    if time.time() < deadline - 5:
         # Try to improve with a very high lambda
         lam_final = 5000.0
         result = opt.minimize(
            objective_and_grad, 
            x, 
            args=(lam_final,),
            jac=True,
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
         if result.fun < objective_and_grad(x, lam_final)[0]:
             x = result.x
             x = np.nan_to_num(x)

    return x

