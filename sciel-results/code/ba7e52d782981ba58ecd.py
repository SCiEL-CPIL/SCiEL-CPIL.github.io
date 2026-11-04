import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
# Safety margin: The limit is 150s. We aim to finish by 140s to avoid hard kill.
DEADLINE_SECONDS = 140.0 

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute triangle indices once for pairwise distances
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)
    
    # Bounds for optimization (slightly inside the cube to avoid boundary issues in gradients)
    bounds = [(-0.4999, 0.4999)] * (N * 3)

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3) or not np.all(np.isfinite(c)):
            return 0.0
        # Check if points are inside the cube (with small tolerance)
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
            
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        g_k are the constraints: pairwise distances / 2 and wall distances.
        """
        c = c_flat.reshape(N, 3)

        # If points are way out, return a large penalty (though bounds should prevent this)
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise distances
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        # Extract upper triangle for efficiency
        d_pairs = d_full[tri_i, tri_j] / 2.0

        # Wall distances (6 walls)
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min value: min(g) approximated by -1/beta * log(sum(exp(-beta*g)))
        neg_beta_g = -beta * g_all
        # Use logsumexp for numerical stability
        lse = logsumexp(neg_beta_g)
        val = -lse / beta

        # Weights for gradient calculation
        weights = np.exp(neg_beta_g - lse)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        # Avoid division by zero
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        
        # Gradient of soft_min w.r.t. g_i is -weights_i
        # Gradient of g_i (pair dist / 2) w.r.t. c_i is -0.5 * unit_vec (pointing away from j? No, d(c_i - c_j)/dc_i = unit_vec)
        # Wait, d = ||c_i - c_j||. d/dc_i (||c_i - c_j||/2) = (c_i - c_j) / (2 * ||c_i - c_j||) = 0.5 * unit_vec.
        # We are minimizing -soft_min. So grad_obj = - grad_soft_min.
        # grad_soft_min = sum( weights_k * grad_g_k )
        # grad_obj = - sum( weights_k * grad_g_k )
        
        # For pair k (i,j):
        # grad_g_k w.r.t c_i = 0.5 * unit_vec_ij
        # grad_g_k w.r.t c_j = -0.5 * unit_vec_ij
        # Contribution to grad_obj w.r.t c_i = - weights_k * 0.5 * unit_vec_ij
        # Contribution to grad_obj w.r.t c_j = - weights_k * (-0.5 * unit_vec_ij) = + weights_k * 0.5 * unit_vec_ij
        
        w_pair = weights[:n_pairs] * 0.5
        contrib = w_pair[:, np.newaxis] * unit_vecs

        # c_i gets -contrib, c_j gets +contrib
        np.add.at(grad_c, tri_i, -contrib)
        np.add.at(grad_c, tri_j, contrib)

        # Wall gradients
        # g_wall = 0.5 - c_x. dg/dc_x = -1.
        # grad_obj w.r.t c_x = - weights * (-1) = + weights
        # g_wall = c_x + 0.5. dg/dc_x = 1.
        # grad_obj w.r.t c_x = - weights * 1 = - weights
        
        ws = n_pairs
        # x_pos: 0.5 - x. grad = +w
        grad_c[:, 0] += weights[ws:ws + N]
        # x_neg: x + 0.5. grad = -w
        grad_c[:, 0] -= weights[ws + N:ws + 2 * N]
        # y_pos
        grad_c[:, 1] += weights[ws + 2 * N:ws + 3 * N]
        # y_neg
        grad_c[:, 1] -= weights[ws + 3 * N:ws + 4 * N]
        # z_pos
        grad_c[:, 2] += weights[ws + 4 * N:ws + 5 * N]
        # z_neg
        grad_c[:, 2] -= weights[ws + 5 * N:ws + 6 * N]

        return val, -grad_c.flatten() # Note: smooth_objective returns val = -soft_min? 
        # Wait, val = -lse/beta. lse = log(sum(exp(-beta g))). 
        # soft_min = -1/beta * log(sum(exp(-beta g))). 
        # So val = soft_min. 
        # We want to MAXIMIZE soft_min. So we MINIMIZE -soft_min.
        # The function returns val (which is soft_min) and grad (which is grad of soft_min).
        # minimize(fun) expects fun to return the value to minimize.
        # So we should return -val and -grad.
        
        # Let's re-verify the sign.
        # Objective to minimize: F = - soft_min(g)
        # dF/dx = - d(soft_min)/dx = - grad_soft_min
        # In the code above, I calculated grad_c as the gradient of the objective to be minimized?
        # Let's trace:
        # w_pair = weights * 0.5
        # contrib = w_pair * unit_vec
        # np.add.at(grad_c, tri_i, -contrib) -> This is - w * 0.5 * u
        # This matches d(-soft_min)/dc_i for the pair term?
        # d(soft_min)/dc_i = sum w_k * d(g_k)/dc_i = sum w_k * 0.5 * u
        # d(-soft_min)/dc_i = - sum w_k * 0.5 * u
        # So grad_c currently holds d(-soft_min)/dc.
        # So we should return -val (which is -soft_min) and grad_c (which is d(-soft_min)/dc).
        
        return -val, grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter):
        """Run L-BFGS-B on the smooth surrogate."""
        x0 = c_init.flatten()
        
        def fun(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val

        def jac(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
        except Exception:
            return c_init.copy()

    def run_direct_optimization(c_init, max_iter):
        """Run L-BFGS-B on the exact non-smooth objective (finite differences)."""
        x0 = c_init.flatten()
        
        def fun(x):
            return -get_radius(x.reshape(N, 3))

        try:
            result = minimize(
                fun, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
        except Exception:
            return c_init.copy()

    def generate_initialization(i):
        """Generate diverse initial guesses."""
        if i % 3 == 0:
            # Random uniform
            return np.random.uniform(-0.4, 0.4, (N, 3))
        elif i % 3 == 1:
            # Perturbed Grid (4x4x4)
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            grid = grid[idx[:N]]
            grid += np.random.normal(0, 0.02, grid.shape)
            return np.clip(grid, -0.49, 0.49)
        else:
            # Rotated Grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            grid = grid[idx[:N]]
            
            A = np.random.normal(0, 1, (3, 3))
            Q, R = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            grid = grid @ Q.T
            max_coord = np.max(np.abs(grid))
            if max_coord > 0.45:
                grid = grid * (0.45 / max_coord)
            return grid

    best_centers = None
    best_score = -1.0
    
    # Aggressive beta schedule from inspiration
    betas = [10, 50, 200, 1000, 5000, 20000, 100000]
    
    # Number of restarts. We have ~140s. 
    # Smooth opt with gradients is fast. 
    # Let's do a few full annealing cycles.
    n_restarts = 5
    
    for r_idx in range(n_restarts):
        if remaining() < 10.0:
            break
            
        c_init = generate_initialization(r_idx)
        c_best = c_init.copy()
        r_best = get_radius(c_init)
        
        # Annealing loop
        for idx, beta in enumerate(betas):
            if remaining() < 5.0:
                break
                
            # Allocate iterations based on remaining time and beta
            # Higher beta needs fewer iterations
            n_remaining_stages = len(betas) - idx
            stage_time = remaining() / n_remaining_stages * 0.8
            max_iter = int(min(300, max(20, stage_time * 20 / (1 + 0.5 * np.log2(beta)))))
            
            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()
        
        # Final direct polish on the exact objective
        if remaining() > 5.0:
            c_polished = run_direct_optimization(c_best, max_iter=100)
            r_polished = get_radius(c_polished)
            if r_polished > r_best:
                r_best = r_polished
                c_best = c_polished.copy()
        
        # Update global best
        if r_best > best_score:
            best_score = r_best
            best_centers = c_best.copy()
            
        # If we hit a very high score, we might stop early to save time?
        # Best known is ~0.13595. 
        if best_score > 0.136:
             # Try one more polish? Or break?
             # Let's break to ensure we return valid result safely.
             break

    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
