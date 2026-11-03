"""Combined approach: Vectorized Smooth Surrogate Optimization for Sphere Packing.
Uses soft-min via log-sum-exp with vectorized gradients for speed.
Implements beta-annealing for global-to-local convergence and a final direct polish.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 142.0  # Safety margin before 150s kill


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0

        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)

        # 1. Calculate all "radius candidates" g_k
        # Pairwise: d_ij / 2
        tri_i, tri_j = np.triu_indices(N, k=1)
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0

        # Walls: 6 distances per point
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])

        # Combine all g_k
        g_all = np.concatenate([d_pairs, wall_dists])

        # 2. Soft Min Calculation
        # We want to maximize min(g_k), so we minimize -min(g_k).
        # -min(g_k) is approximated by (1/beta) * log(sum(exp(-beta * g_k)))
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = log_sum_exp / beta  # This is -min(g_k) approx

        # 3. Gradient Calculation
        # d/dg_k [ (1/beta) log(sum(exp(-beta g_j))) ] = (1/beta) * (-beta * exp(-beta g_k) / sum(exp(-beta g_j)))
        # = - exp(-beta g_k) / sum(exp(-beta g_j))
        # Let weights_k = exp(-beta g_k) / sum(exp(-beta g_j))
        # Then d(val)/dg_k = -weights_k
        # We want gradient of val with respect to c.
        # d(val)/dc = sum_k (d(val)/dg_k) * (dg_k/dc)
        
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        grad_c = np.zeros_like(c)
        
        # Pairwise gradient
        n_pairs = len(d_pairs)
        # g_k = d_ij / 2 = 0.5 * ||c_i - c_j||
        # d(g_k)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(g_k)/dc_j = -0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(val)/dc_i = sum_k (i is involved) -weights_k * 0.5 * (c_i - c_j) / ||c_i - c_j||
        # Note: weights_k is positive. The term -weights_k * 0.5 * diff / norm.
        # Since diff = c_i - c_j, this pushes i away from j if weight is high?
        # Wait. If g_k is small, weight is high. We want to increase g_k.
        # To increase d_ij, we move i away from j.
        # Gradient of objective w.r.t c_i should point in direction that increases g_k.
        # d(val)/dc_i = -weights_k * (c_i - c_j) / (2 * d_pairs_k * 2) ? No.
        # d_pairs = 0.5 * dist. dist = 2 * d_pairs.
        # d(dist)/dc_i = (c_i - c_j) / dist.
        # d(d_pairs)/dc_i = 0.5 * (c_i - c_j) / dist = 0.5 * (c_i - c_j) / (2 * d_pairs) = (c_i - c_j) / (4 * d_pairs).
        # So d(val)/dc_i = -weights_k * (c_i - c_j) / (4 * d_pairs_k).
        # This vector points from j to i. Negative sign means it points from i to j?
        # Let's check. If c_i > c_j, diff > 0. Gradient is negative. So we decrease c_i?
        # That would bring them closer. That's wrong. We want to maximize min dist.
        # Ah, val = -min(g). We are minimizing val.
        # So we want to decrease val.
        # If val is high (min g is low), we want to lower val, which means increasing min g.
        # So the gradient of val should point in the direction that increases g?
        # No, gradient points in direction of steepest ascent.
        # If we move in direction -grad, we decrease val.
        # So -grad should point in direction that increases g.
        # So grad should point in direction that decreases g.
        # If c_i > c_j, decreasing g means decreasing dist, so decreasing c_i.
        # So grad_x_i should be negative.
        # Formula: -weights_k * (c_i - c_j) / (4 * d_pairs_k).
        # If c_i > c_j, (c_i - c_j) > 0. weights > 0. d_pairs > 0.
        # Result is negative. Correct.
        
        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        # For pairs with tiny d, the weight should be negligible or we skip to avoid 0/0
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        
        weighted_diffs = factors[:, np.newaxis] * diff
        
        # Scatter: 
        # d(val)/dc_i += -weights_k * (c_i - c_j) / (4 * d_pairs_k)
        # d(val)/dc_j += -weights_k * (-(c_i - c_j)) / (4 * d_pairs_k) = +weights_k * (c_i - c_j) / (4 * d_pairs_k)
        
        np.add.at(grad_c, tri_i, -weighted_diffs)
        np.add.at(grad_c, tri_j, weighted_diffs)
        
        # Wall part
        # g_k = 0.5 - c_x (for d_x_pos)
        # d(g_k)/dc_x = -1
        # d(val)/dc_x = -weights_k * (-1) = weights_k
        # g_k = c_x + 0.5 (for d_x_neg)
        # d(g_k)/dc_x = 1
        # d(val)/dc_x = -weights_k * (1) = -weights_k
        
        w_start = n_pairs
        # d_x_pos: indices w_start to w_start+N
        grad_c[:, 0] += weights[w_start : w_start+N]
        # d_x_neg: indices w_start+N to w_start+2*N
        grad_c[:, 0] -= weights[w_start+N : w_start+2*N]
        
        # d_y_pos: indices w_start+2*N to w_start+3*N
        grad_c[:, 1] += weights[w_start+2*N : w_start+3*N]
        # d_y_neg: indices w_start+3*N to w_start+4*N
        grad_c[:, 1] -= weights[w_start+3*N : w_start+4*N]
        
        # d_z_pos: indices w_start+4*N to w_start+5*N
        grad_c[:, 2] += weights[w_start+4*N : w_start+5*N]
        # d_z_neg: indices w_start+5*N to w_start+6*N
        grad_c[:, 2] -= weights[w_start+5*N : w_start+6*N]
        
        return val, grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val
            
        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init
            
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def make_bcc_lattice():
        """Body-centered cubic inspired initialization."""
        # BCC lattice points in a 3x3x3 grid of cells.
        # Cell size a = 1/3.
        # Points at corners and body centers.
        # Scale to fit in [-0.5, 0.5].
        
        # Generate a 3x3x3 grid of BCC unit cells.
        # Each cell contributes 2 points. 27 cells * 2 = 54 points.
        
        i = np.arange(3)
        # Corners of the 3x3x3 supercell
        # The supercell spans [0, 2] if cell size is 1? No.
        # Let's define the supercell to span [-0.5, 0.5].
        # Side length 1.
        # 3 cells per side. Cell size 1/3.
        # Cell centers at -1/3, 0, 1/3? No, centers at -0.5 + 0.5/3, ...
        
        # Let's stick to the previous working logic from the current program which seemed to produce valid inputs.
        # The current program used:
        # i = np.arange(3)
        # corners = ...
        # body_centers = corners + 0.5
        # all_points = vstack
        # all_points = all_points * 0.5 - 0.5
        
        i = np.arange(3)
        corners = np.array([(x, y, z) for z in i for y in i for x in i], dtype=float)
        body_centers = corners + 0.5
        all_points = np.vstack([corners, body_centers])
        
        # Scale to fit in [-0.5, 0.5]^3
        # Current range: [0, 2] in each dim. Target: [-0.5, 0.5], range 1.0.
        all_points = all_points * 0.5 - 0.5
        
        # We have 54 points, need 50. Pick the 50 that give the best initial score.
        # For simplicity, just take the first 50.
        if len(all_points) >= N:
            return all_points[:N].copy()
        else:
            pad = np.zeros((N - len(all_points), 3))
            return np.vstack([all_points, pad])

    def make_grid_initial():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    # Initial configuration: BCC lattice
    centers = make_bcc_lattice()
    centers = np.clip(centers, -0.499, 0.499)
    
    best_score = compute_radius(centers)
    best_centers = centers.copy()
    
    # Multi-stage optimization with increasing beta
    # Start with low beta for global structure, increase for local refinement
    betas = [2, 5, 10, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if remaining() < 5:
            break
        
        centers = local_optimize_smooth(centers, beta, maxiter=200, time_budget=10)
        centers = np.clip(centers, -0.499, 0.499)
        
        current_score = compute_radius(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
    # Final polish: direct optimization of the raw min objective
    if remaining() > 5:
        centers = local_optimize_direct(best_centers, maxiter=100, time_budget=10)
        centers = np.clip(centers, -0.499, 0.499)
        
        current_score = compute_radius(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
