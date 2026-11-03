"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp
import random

def construct():
    start_time = time.time()
    # Leave 5s buffer for 150s limit
    time_limit = 145.0 

    def get_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        # Distance to each wall: 0.5 - x, x + 0.5, 0.5 - y, y + 0.5, 0.5 - z, z + 0.5
        # min_i min_k (0.5 - |c_ik|) is equivalent to min(0.5 - max|c_ik|)
        # Actually, min distance to boundary is min(0.5 - c, c + 0.5) for each coordinate
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective_and_grad(c_flat, beta):
        # c_flat is (3N,)
        c = c_flat.reshape(N, 3)
        
        # Calculate distances
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        tri_idx = np.triu_indices(N, k=1)
        d_pairs = d[tri_idx] / 2.0
        
        wall_dists = np.concatenate([
            d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg
        ])
        
        g_all = np.concatenate([d_pairs, wall_dists])
        
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta
        
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        grad_c = np.zeros_like(c)
        
        # Vectorized Pairwise Gradient
        # d_pairs[k] = 0.5 * ||c[i_k] - c[j_k]||
        # d(d_pairs[k])/dc[i_k] = 0.5 * (c[i_k] - c[j_k]) / ||c[i_k] - c[j_k]||
        # d(d_pairs[k])/dc[j_k] = -0.5 * (c[i_k] - c[j_k]) / ||c[i_k] - c[j_k]||
        
        i_idx = tri_idx[0]
        j_idx = tri_idx[1]
        
        # Get differences for all pairs
        # diff_pairs shape (n_pairs, 3)
        diff_pairs = c[i_idx] - c[j_idx]
        
        # Get distances for all pairs
        d_pairs_full = np.linalg.norm(diff_pairs, axis=1)
        
        # Avoid division by zero
        safe_d = np.where(d_pairs_full > 1e-12, d_pairs_full, 1.0)
        
        # Weights for pairs
        w_pairs = weights[:len(d_pairs)]
        
        # Gradient contribution for each pair is w_pairs[k] * 0.5 * diff_pairs[k] / safe_d[k]
        # We need to scatter-add this to c[i_idx] and subtract from c[j_idx]
        
        # grad_contrib shape (n_pairs, 3)
        grad_contrib = (w_pairs[:, np.newaxis] * 0.5 / safe_d[:, np.newaxis]) * diff_pairs
        
        # Accumulate to grad_c
        # Use np.add.at for accumulation
        np.add.at(grad_c, i_idx, grad_contrib)
        np.add.at(grad_c, j_idx, -grad_contrib)
        
        # Wall part
        w_start = len(d_pairs)
        
        # d_x_pos weights: w_start to w_start+N-1
        grad_c[:, 0] -= weights[w_start : w_start+N]
        # d_x_neg weights: w_start+N to w_start+2N-1
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        
        # d_y_pos weights: w_start+2N to w_start+3N-1
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        # d_y_neg weights: w_start+3N to w_start+4N-1
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        
        # d_z_pos weights: w_start+4N to w_start+5N-1
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        # d_z_neg weights: w_start+5N to w_start+6N-1
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # The objective we want to MINIMIZE is -soft_min.
        # So we return -val and -grad_c.
        return -val, -grad_c.flatten()

    def run_l_bfgs(c_init, beta_start, beta_end, max_iter, time_budget):
        # Anneal beta
        n_restarts = 3
        betas = np.linspace(beta_start, beta_end, n_restarts)
        
        best_c = c_init.copy()
        best_r = get_radius(c_init)
        
        for beta in betas:
            # Run L-BFGS-B
            # We need to check time
            if time.time() - start_time > time_budget:
                break
                
            # Flatten
            c_flat = c_init.flatten()
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (3 * N)
            
            # Callback to check time? L-BFGS-B doesn't support time-based stopping easily in all versions.
            # We'll just run it and hope it finishes. If it's too slow, we reduce max_iter.
            # For N=50, L-BFGS-B is quite fast.
            
            try:
                res = opt.minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    c_flat,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
                )
                
                c_new = res.x.reshape(N, 3)
                r_new = get_radius(c_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                
                c_init = c_new.copy()
            except Exception as e:
                print(f"Optimization error: {e}")
                continue
                
        return best_c, best_r

    # 1. Generate Initial Candidates
    
    # Candidate 1: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    
    # Candidate 2: BCC Lattice
    # BCC has 2 atoms per cell.
    # We need 50 atoms. 25 cells.
    # 3x3x3 = 27 cells (54 atoms). Take 50.
    # Or 2x3x3 = 18 cells (36 atoms). Too few.
    # Let's generate a 3x3x3 BCC lattice and take the first 50.
    # BCC lattice points: (0,0,0) and (0.5, 0.5, 0.5) in cell coords.
    # Scale to fit cube.
    # If we use 3x3x3 cells, total size 3x3x3.
    # We want to fit in [-0.5, 0.5].
    # So cell size a = 1/3.
    # Points:
    bcc_points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # Corner
                p1 = np.array([ix, iy, iz]) * (1.0/3.0) - 0.5
                bcc_points.append(p1)
                # Center
                p2 = np.array([ix+0.5, iy+0.5, iz+0.5]) * (1.0/3.0) - 0.5
                bcc_points.append(p2)
    c_bcc = np.array(bcc_points)[:N]
    
    # Candidate 3: Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, (N, 3))
    
    # Candidate 4: Perturbed Grid
    c_grid_pert = c_grid + np.random.uniform(-0.02, 0.02, (N, 3))
    c_grid_pert = np.clip(c_grid_pert, -0.5, 0.5)
    
    # Candidate 5: Perturbed BCC
    c_bcc_pert = c_bcc + np.random.uniform(-0.02, 0.02, (N, 3))
    c_bcc_pert = np.clip(c_bcc_pert, -0.5, 0.5)
    
    candidates = [
        c_grid,
        c_bcc,
        c_rand,
        c_grid_pert,
        c_bcc_pert
    ]
    
    # Filter candidates to ensure they are inside the cube
    valid_candidates = []
    for c in candidates:
        if np.all(c >= -0.5) and np.all(c <= 0.5):
            valid_candidates.append(c)
    
    if not valid_candidates:
        valid_candidates = [c_grid]

    best_overall_c = valid_candidates[0].copy()
    best_overall_r = get_radius(valid_candidates[0])
    
    # 2. Optimization Loop
    # Vectorized gradient allows more iterations.
    # Increase max_iter and add more restarts if time permits.
    
    for i, c_init in enumerate(valid_candidates):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Run optimization
        # Increase max_iter to 1000 due to speedup
        c_opt, r_opt = run_l_bfgs(c_init, beta_start=10.0, beta_end=500.0, max_iter=1000, time_budget=time_limit - 10)
        
        if r_opt > best_overall_r:
            best_overall_r = r_opt
            best_overall_c = c_opt.copy()
            
        # Optional: Perturb the result and run again if time permits
        if time.time() - start_time < time_limit - 20 and i < 3:
            # Perturb
            c_pert = c_opt + np.random.normal(0, 0.01, (N, 3))
            c_pert = np.clip(c_pert, -0.5, 0.5)
            c_opt2, r_opt2 = run_l_bfgs(c_pert, beta_start=50.0, beta_end=1000.0, max_iter=500, time_budget=time_limit - 10)
            if r_opt2 > best_overall_r:
                best_overall_r = r_opt2
                best_overall_c = c_opt2.copy()

    # 3. Final Polish
    # If we have time left, do a high-beta run on the best candidate
    if time.time() - start_time < time_limit - 5:
        c_polish = best_overall_c + np.random.normal(0, 0.001, (N, 3))
        c_polish = np.clip(c_polish, -0.5, 0.5)
        # Use very high beta
        c_final, r_final = run_l_bfgs(c_polish, beta_start=1000.0, beta_end=5000.0, max_iter=500, time_budget=time_limit - 2)
        if r_final > best_overall_r:
            best_overall_r = r_final
            best_overall_c = c_final.copy()

    return best_overall_c

# EVOLVE-BLOCK-END
