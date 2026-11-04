"""
Circle packing in unit square using a combinatorial gap-filling strategy.
1. Initialize with a dense grid of candidate centers.
2. Iteratively select the best location for the next circle (maximizing minimum distance to existing circles and boundaries).
3. Use local search (swaps and moves) to improve the arrangement.
4. Compute radii and validate.
This avoids the local minima traps of continuous gradient descent on the soft-min objective.
"""
import numpy as np
from scipy.spatial import Delaunay, Voronoi
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def score_config(x, y):
        r = compute_radii(x, y)
        if check_validity(x, y, r):
            return np.sum(r)
        return -np.inf

    def greedy_packing(n_circles, seed=0):
        """
        Greedy algorithm: place circles one by one at the location that maximizes
        the minimum distance to existing circles and boundaries.
        Uses a fine grid for candidate locations.
        """
        rng = np.random.RandomState(seed)
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        
        # Candidate grid: 100x100 points
        grid_size = 100
        gx = np.linspace(0.02, 0.98, grid_size)
        gy = np.linspace(0.02, 0.98, grid_size)
        Gx, Gy = np.meshgrid(gx, gy)
        Gx = Gx.flatten()
        Gy = Gy.flatten()
        
        # Precompute boundary distances for grid
        r_bound_grid = np.minimum(np.minimum(Gx, 1.0 - Gx), np.minimum(Gy, 1.0 - Gy))
        
        for i in range(n_circles):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Calculate distance from each grid point to all existing circles
            if i == 0:
                # For the first circle, just pick the center
                x[i] = 0.5
                y[i] = 0.5
                continue
                
            # Vectorized distance calculation
            dx = Gx[:, None] - x[:i][None, :]
            dy = Gy[:, None] - y[:i][None, :]
            dists = np.sqrt(dx**2 + dy**2)
            min_dist_to_existing = np.min(dists, axis=1)
            
            # The "radius" a circle at this grid point could have
            potential_r = np.minimum(r_bound_grid, 0.5 * min_dist_to_existing)
            
            # Add some randomness to break ties and explore
            noise = rng.uniform(0, 0.005, len(Gx))
            potential_r += noise
            
            # Pick the best location
            best_idx = np.argmax(potential_r)
            x[i] = Gx[best_idx]
            y[i] = Gy[best_idx]
            
        return x, y

    def local_search_improve(x, y, max_iters=5000):
        """
        Improve the configuration by moving circles to better positions.
        Uses a simple hill-climbing approach with random restarts for stuck moves.
        """
        x = x.copy()
        y = y.copy()
        current_score = score_config(x, y)
        
        # If initial score is bad, try to fix validity first by shrinking
        r = compute_radii(x, y)
        if not check_validity(x, y, r):
             # This shouldn't happen with greedy, but just in case
             return x, y, current_score

        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]
        no_improve_count = 0
        
        for step_idx, step in enumerate(step_sizes):
            for _ in range(max_iters // (len(step_sizes) * (step_idx+1))):
                if time.time() - start_time > DEADLINE - 5:
                    return x, y, current_score
                
                # Pick a random circle
                i = np.random.randint(N)
                
                # Try moving it in a random direction
                angle = np.random.uniform(0, 2 * np.pi)
                new_x = x[i] + step * np.cos(angle)
                new_y = y[i] + step * np.sin(angle)
                
                # Check boundaries
                if new_x < 0.01 or new_x > 0.99 or new_y < 0.01 or new_y > 0.99:
                    continue
                
                x[i] = new_x
                y[i] = new_y
                
                new_score = score_config(x, y)
                
                if new_score > current_score + 1e-9:
                    current_score = new_score
                    no_improve_count = 0
                else:
                    # Revert
                    x[i] -= step * np.cos(angle)
                    y[i] -= step * np.sin(angle)
                    no_improve_count += 1
                    if no_improve_count > 100:
                        break
                        
        return x, y, current_score

    def swap_improve(x, y, max_iters=2000):
        """
        Try swapping positions of pairs of circles to improve packing.
        """
        x = x.copy()
        y = y.copy()
        current_score = score_config(x, y)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            i, j = np.random.choice(N, 2, replace=False)
            
            # Swap
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]
            
            new_score = score_config(x, y)
            
            if new_score > current_score + 1e-9:
                current_score = new_score
            else:
                # Revert
                x[i], x[j] = x[j], x[i]
                y[i], y[j] = y[j], y[i]
                
        return x, y, current_score

    # Strategy: Multiple starts with different greedy seeds and local search refinements
    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None

    # Run several greedy starts
    for seed in range(10):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        x, y = greedy_packing(N, seed)
        
        # Local search polish
        x, y, score = local_search_improve(x, y, max_iters=2000)
        
        # Swap polish
        x, y, score = swap_improve(x, y, max_iters=500)
        
        # Final local search
        x, y, score = local_search_improve(x, y, max_iters=1000)
        
        r = compute_radii(x, y)
        
        if not check_validity(x, y, r):
            continue
            
        if score > best_score:
            best_score = score
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()

    # If we haven't found a good solution, fall back to a simple grid
    if best_x is None:
        # Simple 5x5 grid + 1
        x = []
        y = []
        for i in range(5):
            for j in range(5):
                x.append((j + 0.5) / 5.0)
                y.append((i + 0.5) / 5.0)
        x.append(0.5)
        y.append(0.5)
        x = np.array(x)
        y = np.array(y)
        best_r = compute_radii(x, y)
        best_x = x
        best_y = y
        best_score = np.sum(best_r)

    # Final high-precision polish using scipy on the best solution found
    if time.time() - start_time < DEADLINE - 10:
        # Use the soft-min objective from the original strategy but only for the final polish
        # This combines the robustness of the combinatorial init with the precision of gradient descent
        
        def soft_objective_and_grad(params, beta):
            x = params[:N]
            y = params[N:2*N]

            b_x = np.minimum(x, 1.0 - x)
            b_y = np.minimum(y, 1.0 - y)
            r_bound = np.minimum(b_x, b_y)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-30)
            np.fill_diagonal(dist, 1e10)

            c = 0.5 * dist
            np.fill_diagonal(c, r_bound)

            neg_beta_c = -beta * c
            max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
            exp_terms = np.exp(neg_beta_c - max_neg)
            sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
            log_sum = np.log(sum_exp)
            r = -(max_neg + log_sum) / beta
            r = r[:, 0]

            w = exp_terms / sum_exp

            dist_safe = dist.copy()
            np.fill_diagonal(dist_safe, 1.0)

            dc_dx_i = 0.5 * dx / dist_safe
            dc_dy_i = 0.5 * dy / dist_safe
            dc_dx_j = -0.5 * dx / dist_safe
            dc_dy_j = -0.5 * dy / dist_safe

            beta_b = 200.0
            bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
            neg_bb = -beta_b * bounds_4
            max_bb = np.max(neg_bb, axis=1, keepdims=True)
            exp_bb = np.exp(neg_bb - max_bb)
            sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
            w_b = exp_bb / sum_bb

            grad_rb_dx = w_b[:, 0] - w_b[:, 1]
            grad_rb_dy = w_b[:, 2] - w_b[:, 3]

            np.fill_diagonal(dc_dx_i, grad_rb_dx)
            np.fill_diagonal(dc_dy_i, grad_rb_dy)
            np.fill_diagonal(dc_dx_j, grad_rb_dx)
            np.fill_diagonal(dc_dy_j, grad_rb_dy)

            w_dc_dx_j = w * dc_dx_j
            w_dc_dx_i = w * dc_dx_i
            w_dc_dy_j = w * dc_dy_j
            w_dc_dy_i = w * dc_dy_i

            term1_dx = np.sum(w_dc_dx_j, axis=0)
            term2_dx = np.sum(w_dc_dx_i, axis=1) - np.diag(w) * np.diag(dc_dx_i)
            dr_dx = term1_dx + term2_dx

            term1_dy = np.sum(w_dc_dy_j, axis=0)
            term2_dy = np.sum(w_dc_dy_i, axis=1) - np.diag(w) * np.diag(dc_dy_i)
            dr_dy = term1_dy + term2_dy

            grad = np.concatenate([-dr_dx, -dr_dy])
            obj_val = -np.sum(r)
            return obj_val, grad

        params0 = np.concatenate([best_x, best_y])
        bounds = [(0.005, 0.995)] * (2 * N)
        
        try:
            from scipy.optimize import minimize
            res = minimize(soft_objective_and_grad, params0,
                           method='L-BFGS-B', bounds=bounds,
                           args=(500.0,),
                           jac=True,
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            
            x_polish = res.x[:N]
            y_polish = res.x[N:]
            r_polish = compute_radii(x_polish, y_polish)
            
            if check_validity(x_polish, y_polish, r_polish):
                score_polish = np.sum(r_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_x = x_polish
                    best_y = y_polish
                    best_r = r_polish
        except Exception:
            pass

    # Ensure validity one last time
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
