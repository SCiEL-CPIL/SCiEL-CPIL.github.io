"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    # 1. Initialization: Hexagonal Grid
    # We want to fit N=200 circles.
    # Hexagonal packing density is sqrt(3)/2 * (1/r)^2 ... 
    # Let's estimate spacing s. Area per circle ~ s^2 * sqrt(3)/2.
    # N * (s^2 * sqrt(3)/2) ~ 1.0 => s ~ sqrt(2 / (N * sqrt(3)))
    s = math.sqrt(2.0 / (N * math.sqrt(3.0)))
    
    # Generate hexagonal points
    points = []
    # Number of rows
    # Width ~ cols * s
    # Height ~ rows * s * sqrt(3)/2
    # Let's try to fit as many as possible and then trim/pad
    max_coords = 10
    for row in range(-max_coords, max_coords + 1):
        for col in range(-max_coords, max_coords + 1):
            x = col * s
            y = row * s * math.sqrt(3.0) / 2.0
            if row % 2 != 0:
                x += s / 2.0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append((x, y))
    
    # If we have more than N, we need to select N.
    # If we have less, we need to add.
    # Simple strategy: Keep the ones closest to the center? Or just take the first N?
    # Better: Sort by distance to center, take N closest? 
    # Actually, for packing, we want them spread out. 
    # If we have too many, we can just take a subset that maintains the lattice structure roughly.
    # If we have too few, we can add points in the largest gaps.
    
    # Let's refine the generation to get exactly N or close.
    # For N=200, s ~ 0.075.
    # Square side 1. 
    # Cols ~ 1/0.075 = 13.3. Rows ~ 1/(0.075*0.866) = 15.4.
    # 13 * 15 = 195. Close.
    
    # Let's just generate a grid of candidates and pick the best N via a simple local relaxation or just take the first N from a centered spiral?
    # Taking first N from a centered list is good.
    
    points_arr = np.array(points)
    if len(points_arr) > N:
        # Sort by distance to center
        dists = np.sum(points_arr**2, axis=1)
        idx = np.argsort(dists)
        centers = points_arr[idx[:N]].copy()
    elif len(points_arr) < N:
        # This shouldn't happen with max_coords=10, but just in case
        # Pad with random points near center
        centers = np.zeros((N, 2))
        if len(points_arr) > 0:
             centers[:len(points_arr)] = points_arr
        # The rest are 0,0 which is bad. Let's just generate more rows/cols if needed.
        # But max_coords=10 gives ~400 points. So len > N is likely.
    
    # 2. Optimization using L-BFGS-B with analytic gradient
    
    def compute_radius_and_grad(c):
        """
        Computes the minimum radius and its gradient.
        c: (N, 2) array.
        Returns: radius, gradient (N, 2)
        """
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        # dist to boundary = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
        # For a point (x, y):
        # d_x_pos = 0.5 - x
        # d_x_neg = 0.5 + x
        # d_y_pos = 0.5 - y
        # d_y_neg = 0.5 + y
        # b_i = min of these 4.
        
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        
        min_pair_dist = np.min(dists)
        min_boundary_dist = np.min(b)
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        grad = np.zeros_like(c)
        
        # Determine which constraint is active
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Find the pair with min distance
            # There might be multiple. We pick one arbitrarily (or sum gradients? Sum is safer for non-uniqueness)
            # For simplicity, find all pairs with dist close to min
            # But usually one dominant pair.
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i == idx_j:
                 # Should not happen due to inf diagonal
                 pass
            else:
                d = dists[idx_i, idx_j]
                # Gradient of d_ij / 2
                # d(d_ij)/dx_i = (x_i - x_j) / d_ij
                # d(d_ij)/dx_j = (x_j - x_i) / d_ij
                vec = (c[idx_i] - c[idx_j]) / d
                grad[idx_i] += vec / 2.0
                grad[idx_j] -= vec / 2.0
        else:
            # Boundary constraint active
            # Find point(s) with min boundary distance
            min_b = np.min(b)
            # Find all points with b == min_b (within tolerance)
            active_idx = np.where(np.isclose(b, min_b, atol=1e-9))[0]
            for i in active_idx:
                x, y = c[i]
                # Determine which boundary is closest
                d_pos_x = 0.5 - x
                d_neg_x = 0.5 + x
                d_pos_y = 0.5 - y
                d_neg_y = 0.5 + y
                
                # Gradient of min(d_pos_x, d_neg_x, d_pos_y, d_neg_y)
                # If x > 0, min is d_pos_x, grad is (-1, 0)
                # If x < 0, min is d_neg_x, grad is (1, 0)
                # If y > 0, min is d_pos_y, grad is (0, -1)
                # If y < 0, min is d_neg_y, grad is (0, 1)
                
                # Actually, we just need the direction to move INWARDS.
                # If x > 0, move left (-1, 0)
                # If x < 0, move right (1, 0)
                # If y > 0, move down (0, -1)
                # If y < 0, move up (0, 1)
                
                # But we need to know WHICH boundary is the min for this point.
                # b_i = min(0.5-|x|, 0.5-|y|)
                # If 0.5-|x| < 0.5-|y|, then x is the limiting factor.
                # Grad of (0.5 - |x|) is -sign(x) in x direction.
                # Grad of (0.5 - |y|) is -sign(y) in y direction.
                
                if 0.5 - abs(x) <= 0.5 - abs(y):
                    # x is limiting
                    if x > 0:
                        grad[i, 0] += -1.0
                    elif x < 0:
                        grad[i, 0] += 1.0
                    else:
                        # x=0, ambiguous, pick 0 or average?
                        pass
                else:
                    # y is limiting
                    if y > 0:
                        grad[i, 1] += -1.0
                    elif y < 0:
                        grad[i, 1] += 1.0
                    else:
                        pass
                        
        return radius, grad

    def objective_and_grad(c_flat):
        c = c_flat.reshape(N, 2)
        r, g = compute_radius_and_grad(c)
        # We want to MAXIMIZE r, so MINIMIZE -r
        return -r, -g.flatten()

    # Run optimization
    best_radius = compute_radius_and_grad(centers)[0]
    best_centers = centers.copy()
    
    # L-BFGS-B setup
    # maxiter should be high, but we check time.
    # We will run multiple rounds with decreasing "tolerance" or just let it run.
    
    # To handle non-smoothness better, we can add a small random perturbation if stuck.
    # But L-BFGS-B is quite robust.
    
    # Let's use a soft-min surrogate for the first phase to smooth the landscape, 
    # then switch to exact min for the final polish.
    
    def soft_objective_and_grad(c_flat, tau):
        c = c_flat.reshape(N, 2)
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e-9)
        
        # Boundary
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        
        # Soft min of all distances / 2
        # vals = np.concatenate([dists.ravel()/2, b])
        # This is expensive to build.
        
        # We can compute log-sum-exp efficiently.
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        # soft_min(x) = -LSE(-x)
        
        # Let's just use the exact min for now, it's simpler and often works if init is good.
        # If it fails, we can add noise.
        
        r, g = compute_radius_and_grad(c)
        return -r, -g.flatten()

    # Optimization Loop
    # We will run L-BFGS-B for the available time.
    # Since L-BFGS-B doesn't have a time limit built-in, we run it in chunks?
    # Or just set maxiter high and hope it finishes or converges.
    # Better: Run it, and if it takes too long, we can't interrupt easily.
    # However, for N=200, one L-BFGS-B run usually takes < 1s if well-behaved.
    # So we can run multiple restarts/improvements.
    
    # Strategy:
    # 1. Run L-BFGS-B on current centers.
    # 2. If time remains, perturb centers slightly and run again.
    
    def run_optimize(c0, max_time_per_run=5.0):
        start = time.time()
        res = minimize(objective_and_grad, c0.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        # Check if it finished quickly
        elapsed = time.time() - start
        return res

    # First pass
    res = run_optimize(centers)
    if res.success or res.status == 0:
        new_centers = res.x.reshape(N, 2)
        new_radius = compute_radius_and_grad(new_centers)[0]
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers
    
    # Second pass: Perturb and optimize
    # We can try to "kick" the system out of local minima.
    # A good kick: identify the "tight" constraints and push them apart.
    # Or just random perturbations.
    
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        # Try a few random restarts from the best found
        for i in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb best_centers
            noise_scale = 0.001 * (1 - i/10.0)
            perturbed = best_centers + noise_scale * np.random.randn(N, 2)
            # Clip to square
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            
            res = run_optimize(perturbed)
            new_centers = res.x.reshape(N, 2)
            new_radius = compute_radius_and_grad(new_centers)[0]
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers
                # If we found a better one, we can continue from here
                # But let's keep the global best for restarts

    # Final Polish: Very tight tolerance
    if time.time() < deadline - 5:
        res = minimize(objective_and_grad, best_centers.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
        new_centers = res.x.reshape(N, 2)
        new_radius = compute_radius_and_grad(new_centers)[0]
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers

    # Ensure validity: Clip to square strictly
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
