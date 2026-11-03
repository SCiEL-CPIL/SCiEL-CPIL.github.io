import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def _time_left():
        return deadline - time.time()

    def _compute_radius(centers):
        """Compute the exact radius for a given set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        # For 200 points, this is 200*199/2 = 19900 pairs, manageable
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def _soft_min_neg(centers, r_target, beta=50.0):
        """
        Objective: minimize -soft_min(clearances).
        Clearances are:
        1. Wall distances: 0.5 - |c_i| for each i, dim
        2. Pair distances / 2 - r_target for each pair i<j
        
        We want to maximize the minimum clearance.
        soft_min(x) = -(1/beta) * log(sum(exp(-beta * x)))
        So we minimize -soft_min(x) = (1/beta) * log(sum(exp(-beta * x)))
        
        Actually, let's define the objective as:
        f = (1/beta) * log( sum(exp(-beta * c_k)) )
        where c_k are the clearances.
        If all c_k > r_target, and we want to push them higher, this function is small.
        As c_k decreases, exp(-beta*c_k) increases, so f increases.
        We want to minimize f.
        
        Wait, standard trick:
        To maximize min(c_k), we can minimize -min(c_k).
        Approximate -min(c_k) with (1/beta) * log(sum(exp(-beta * c_k))).
        Let's call this J(centers, beta).
        Gradient descent on J will push centers to increase the smallest c_k.
        
        Clearances c_k:
        - For walls: c_{i,d} = 0.5 - |c_{i,d}|
        - For pairs: c_{ij} = ||c_i - c_j|| / 2.0
        
        We don't necessarily need r_target in the objective if we are just maximizing the min clearance.
        However, to "inflate", we can subtract a target radius from the clearances?
        No, simpler: just maximize the min clearance.
        
        Let's implement J(centers, beta) = (1/beta) * log( sum_{k} exp(-beta * c_k) )
        
        Gradient:
        dJ/dc_k = -exp(-beta * c_k) / sum(exp(-beta * c_k))
        dc_k/dx_i ...
        
        This might be numerically unstable for large beta.
        Alternative: Use scipy's L-BFGS-B on the objective J.
        """
        # Wall terms
        # c_wall[i,d] = 0.5 - abs(c[i,d])
        # Let's just compute the values first to see if we need grad.
        # L-BFGS-B supports analytic gradients.
        
        # Vectorized computation of clearances
        # Walls
        x = centers[:, 0]
        y = centers[:, 1]
        
        c_walls = np.concatenate([
            0.5 - np.abs(x),
            0.5 - np.abs(y)
        ])
        
        # Pairs
        # Only upper triangle
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        c_pairs = dists[iu] / 2.0
        
        c_all = np.concatenate([c_walls, c_pairs])
        
        # For numerical stability in log-sum-exp
        # J = (1/beta) * log(sum(exp(-beta * c)))
        # = (1/beta) * ( -beta * max(c) + log(sum(exp(-beta * (c - max(c))))) )
        # = -max(c) + (1/beta) * log(sum(exp(-beta * (c - max(c)))))
        
        c_max = np.max(c_all)
        shifted = -beta * (c_all - c_max)
        # Clip shifted to avoid overflow in exp (shouldn't happen if c_max is max, so shifted <= 0)
        # But if c_all has huge negative values, shifted is huge positive? No, c_max is max, so c - c_max <= 0.
        # So -beta * (c - c_max) >= 0.
        # If c is very negative, exp(-beta * (c - c_max)) is huge.
        # We should probably clip c_all from below or handle carefully.
        # Given the geometry, clearances shouldn't be wildly negative unless points are far outside.
        # We will clip centers in the optimizer callback or bounds.
        
        exp_shifted = np.exp(np.clip(shifted, -700, 700))
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            val = -c_max
            grad = np.zeros_like(centers)
            return val, grad
            
        log_sum_exp = np.log(sum_exp)
        val = -c_max + (1.0 / beta) * log_sum_exp
        
        # Gradient
        # p_k = exp(-beta * (c_k - c_max)) / sum_exp
        p = exp_shifted / sum_exp
        
        grad = np.zeros_like(centers)
        
        # Walls gradient
        # c_walls[0:N] are for x, c_walls[N:2N] are for y
        # d/dx_i of c_walls_x = -sign(x_i)
        # d/dy_i of c_walls_y = -sign(y_i)
        
        # Contribution from walls
        # sum_{i} p_{wall_x_i} * d(c_wall_x_i)/dx_i
        # sum_{i} p_{wall_y_i} * d(c_wall_y_i)/dy_i
        
        # Note: c_all is [c_walls_x, c_walls_y, c_pairs]
        p_walls_x = p[:N]
        p_walls_y = p[N:2*N]
        
        # dc/dx for x-walls: -sign(x)
        # dc/dy for y-walls: -sign(y)
        grad[:, 0] -= p_walls_x * np.sign(centers[:, 0])
        grad[:, 1] -= p_walls_y * np.sign(centers[:, 1])
        
        # Pairs gradient
        # c_pair_ij = ||c_i - c_j|| / 2
        # d/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
        # d/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
        
        # p_pairs corresponds to upper triangle indices
        # We need to accumulate gradients
        # Let's iterate or vectorize
        # For each pair (i, j) with i < j:
        # grad[i] += p_pair * (c_i - c_j) / (2 * dist_ij)
        # grad[j] += p_pair * (c_j - c_i) / (2 * dist_ij)
        
        # dists[iu] are the distances.
        # diff[iu] are the vectors c_i - c_j.
        vecs = diff[iu]
        dists_pairs = dists[iu]
        
        # Avoid division by zero
        mask = dists_pairs > 1e-12
        vecs_safe = np.where(mask[:, np.newaxis], vecs, 0.0)
        dists_safe = np.where(mask, dists_pairs, 1.0)
        
        # Normalized vectors: (c_i - c_j) / (2 * dist)
        norm_vecs = vecs_safe / (2.0 * dists_safe[:, np.newaxis])
        
        # Weighted by p_pairs
        weighted_vecs = norm_vecs * p[2*N:][:, np.newaxis]
        
        # Accumulate
        # For each pair index k, we have i = iu[0][k], j = iu[1][k]
        # grad[i] += weighted_vecs[k]
        # grad[j] -= weighted_vecs[k]  (since vec is c_i - c_j, grad[j] gets -vec)
        
        # Use np.add.at for efficient accumulation
        np.add.at(grad, iu[0], weighted_vecs)
        np.add.at(grad, iu[1], -weighted_vecs)
        
        return val, grad

    def _objective_and_grad(x_flat, beta):
        centers = x_flat.reshape(-1, 2)
        val, grad = _soft_min_neg(centers, beta=beta)
        return val, grad.flatten()

    def _initial_hex_grid(n_rows, n_cols):
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts

    def _square_grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # 1. Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0
    candidates = []

    # Try multiple hexagonal layouts
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = _initial_hex_grid(n_rows, n_cols)
        if pts is not None:
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()
            candidates.append(pts.copy())

    # Also try square grid
    grid = _square_grid_init()
    r_grid = _compute_radius(grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_centers = grid.copy()
    candidates.append(grid.copy())
    
    # Add a random perturbation to the best hex grid for diversity
    if best_centers is not None:
        rand_init = best_centers + np.random.randn(N, 2) * 0.01
        rand_init = np.clip(rand_init, -0.49, 0.49)
        candidates.append(rand_init)

    # 2. Optimize each candidate using L-BFGS-B with soft-min objective
    # We will run optimization for a few of the best candidates
    # Sort candidates by initial radius
    cand_radii = [_compute_radius(c) for c in candidates]
    sorted_idx = np.argsort(cand_radii)[::-1]
    
    max_candidates_to_optimize = 3  # Limit to 3 to stay within time budget
    
    for idx in sorted_idx[:max_candidates_to_optimize]:
        if _time_left() < 30.0:
            break
            
        init_centers = candidates[idx].copy()
        x0 = init_centers.flatten()
        
        # Phase 1: Coarse optimization with low beta
        # L-BFGS-B is good for smooth functions
        # We want to minimize the soft-min of clearances
        # Lower beta = smoother, easier to optimize, but less precise min
        # Higher beta = sharper, harder to optimize, but more precise
        
        current_best = init_centers.copy()
        current_r = _compute_radius(current_best)
        
        # Try a few beta values, increasing sharpness
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if _time_left() < 15.0:
                break
                
            res = minimize(
                _objective_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e10:
                new_centers = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_centers)
                
                if new_r > current_r:
                    current_r = new_r
                    current_best = new_centers.copy()
                    x0 = new_centers.flatten()
                else:
                    # If it didn't improve, maybe stick with previous x0 for next beta?
                    # Or restart from current_best?
                    # Let's restart from current_best to ensure we are building on progress
                    x0 = current_best.flatten()
                    
        # Phase 2: Inflation / Local refinement
        # Now that we are in a local minimum of the soft-min, we can try to push the radius up.
        # One way: fix the objective structure but add a penalty if any clearance < current_r + epsilon?
        # Simpler: Just run L-BFGS-B with very high beta to sharpen the minimum.
        
        for beta in [500.0, 1000.0]:
            if _time_left() < 10.0:
                break
            
            res = minimize(
                _objective_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < 1e10:
                new_centers = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_centers)
                
                if new_r > current_r:
                    current_r = new_r
                    current_best = new_centers.copy()
                    x0 = new_centers.flatten()
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = current_best.copy()

    # 3. Final polish with very high beta on the best result
    if best_centers is not None and _time_left() > 5.0:
        x0 = best_centers.flatten()
        for beta in [2000.0, 5000.0]:
            if _time_left() < 3.0:
                break
            res = minimize(
                _objective_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 20, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success or res.fun < 1e10:
                new_centers = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_centers)
                if new_r > best_radius:
                    best_radius = new_r
                    best_centers = new_centers.copy()
                    x0 = new_centers.flatten()

    if best_centers is None:
        best_centers = _square_grid_init()

    return best_centers
