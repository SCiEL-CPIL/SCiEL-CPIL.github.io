import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit
    
    best_val = -np.inf
    best_solution = None
    
    # Number of restarts
    # We will run as many as possible within the time limit
    max_restarts = 50
    
    # Initial layout generator
    def get_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Create a grid-like distribution with jitter
        # 5x5 grid = 25 points. Add 1 more.
        # Or 6x5 = 30 points, pick 26.
        # Let's use a 5x5 grid and add one in the middle of a random cell, or just jitter a 5x5 grid and add one extra randomly.
        
        # Better: Hexagonal packing approximation
        # Or just random points in [0.1, 0.9]^2
        # Random points often get stuck in local minima.
        # Grid is better.
        
        side = 5
        points = []
        for i in range(side):
            for j in range(side):
                # Center of cell
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add jitter
                x += rng.uniform(-0.1, 0.1)
                y += rng.uniform(-0.1, 0.1)
                points.append([x, y])
        
        # We have 25 points. Need 1 more.
        # Place the 26th point near the center of the square, or random.
        # Let's place it at a random position, maybe near the center to fill gaps.
        points.append([0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1)])
        
        pts = np.array(points[:N])
        
        # Ensure points are within bounds
        pts[:, 0] = np.clip(pts[:, 0], 0.05, 0.95)
        pts[:, 1] = np.clip(pts[:, 1], 0.05, 0.95)
        
        # Initial radii: small
        radii = np.full(N, 0.05)
        
        # Vectorize: [x, y, r]
        v0 = np.zeros((N, 3))
        v0[:, 0] = pts[:, 0]
        v0[:, 1] = pts[:, 1]
        v0[:, 2] = radii
        
        return v0.flatten()

    def objective_and_grad(v):
        """
        Returns objective value and gradient for L-BFGS-B.
        Objective: -sum(r) + lambda * sum(violations^2)
        """
        v = v.reshape(N, 3)
        x = v[:, 0]
        y = v[:, 1]
        r = v[:, 2]
        
        # Penalty coefficient
        # Needs to be large enough to enforce constraints but not dominate the objective initially.
        # Adaptive penalty could be used, but let's start with a fixed one.
        # If constraints are violated, we want to fix them.
        # A value like 100-1000 usually works.
        lam = 100.0
        
        obj = -np.sum(r)
        grad_r = -np.ones(N)
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # 1. Boundary Constraints
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        # Violation: max(0, r - x), max(0, r - (1-x)), etc.
        
        # Left: r - x > 0
        viol_left = np.maximum(0, r - x)
        obj += lam * np.sum(viol_left**2)
        grad_r += 2 * lam * viol_left
        grad_x -= 2 * lam * viol_left
        
        # Right: r - (1-x) = r - 1 + x > 0
        viol_right = np.maximum(0, r - 1 + x)
        obj += lam * np.sum(viol_right**2)
        grad_r += 2 * lam * viol_right
        grad_x += 2 * lam * viol_right
        
        # Bottom: r - y > 0
        viol_bottom = np.maximum(0, r - y)
        obj += lam * np.sum(viol_bottom**2)
        grad_r += 2 * lam * viol_bottom
        grad_y -= 2 * lam * viol_bottom
        
        # Top: r - (1-y) = r - 1 + y > 0
        viol_top = np.maximum(0, r - 1 + y)
        obj += lam * np.sum(viol_top**2)
        grad_r += 2 * lam * viol_top
        grad_y += 2 * lam * viol_top
        
        # 2. Pairwise Constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Violation: (r_i + r_j) - dist_ij > 0
        # dist_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        
        # Compute distance matrix
        # To save memory/time for N=26, we can loop, but vectorization is faster.
        # NxN distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12) # epsilon for stability
        
        r_sum = r[:, None] + r[None, :]
        
        # We only care about i < j
        # Create a mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Distances and r_sums for i < j
        d_ij = dist[mask]
        r_ij = r_sum[mask]
        
        # Violation: r_ij - d_ij
        viol_pair = np.maximum(0, r_ij - d_ij)
        
        obj += lam * np.sum(viol_pair**2)
        
        # Gradients for pairwise
        # d(obj)/d(r_i) += 2 * lam * viol_pair * (d(r_ij)/d(r_i)) = 2 * lam * viol_pair * 1
        # d(obj)/d(x_i) += 2 * lam * viol_pair * (d(r_ij - d_ij)/d(x_i)) = 2 * lam * viol_pair * (- d(d_ij)/d(x_i))
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        
        # We need to scatter the gradients back.
        # This is slightly tricky with masks.
        
        # Let's compute the full matrix of gradients for pairs and then sum.
        # grad_r_pair[i] = sum_{j != i} 2 * lam * viol_ij
        # grad_x_pair[i] = sum_{j != i} 2 * lam * viol_ij * (-(x_i - x_j)/d_ij)
        
        # Recalculate viol_pair as full matrix for easier gradient accumulation?
        # N=26 is small, full matrix is 26x26 = 676 entries. Very fast.
        
        viol_pair_full = np.maximum(0, r_sum - dist)
        # Zero out diagonal
        np.fill_diagonal(viol_pair_full, 0)
        
        # grad_r contribution
        grad_r += 2 * lam * np.sum(viol_pair_full, axis=1)
        
        # grad_x contribution
        # term_ij = 2 * lam * viol_ij * (-(x_i - x_j) / d_ij)
        # Note: dist is symmetric.
        # We need to be careful with division by zero. dist > 0 due to epsilon.
        
        # grad_x_i = sum_j 2 * lam * viol_ij * (x_j - x_i) / d_ij
        # Let's compute the matrix of coefficients
        # coeff_ij = 2 * lam * viol_ij / d_ij
        # Then grad_x_i = sum_j coeff_ij * (x_j - x_i)
        
        coeff = 2 * lam * viol_pair_full / dist
        # coeff is NxN.
        
        # grad_x_i = sum_j coeff_ij * x_j - sum_j coeff_ij * x_i
        # = (coeff * x)[i].sum() - x_i * (coeff.sum(axis=1))[i]
        
        grad_x += np.sum(coeff * x[None, :], axis=1) - x * np.sum(coeff, axis=1)
        
        # Similarly for y
        grad_y += np.sum(coeff * y[None, :], axis=1) - y * np.sum(coeff, axis=1)
        
        # Assemble gradient
        grad = np.zeros_like(v)
        grad[:, 0] = grad_x
        grad[:, 1] = grad_y
        grad[:, 2] = grad_r
        
        return obj, grad.flatten()

    # Run restarts
    for k in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        v0 = get_initial_guess(k)
        
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        try:
            res = minimize(
                objective_and_grad, 
                v0, 
                jac=True, # We return (f, g)
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_val:
                # Check validity
                v_sol = res.x.reshape(N, 3)
                x, y, r = v_sol[:, 0], v_sol[:, 1], v_sol[:, 2]
                
                # Validate constraints strictly
                valid = True
                # Walls
                if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or \
                   np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
                    valid = False
                
                # Pairs
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_solution = v_sol.copy()
                        
        except Exception as e:
            # print(f"Restart {k} failed: {e}")
            pass

    if best_solution is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
