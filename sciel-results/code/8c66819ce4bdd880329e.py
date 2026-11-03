import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Initial position: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()

    def compute_radius(p):
        # p is Nx3
        # Check validity
        if not np.all(np.isfinite(p)):
            return -1.0
        if np.any(p < -0.5) or np.any(p > 0.5):
            return -1.0

        # Wall distance
        wall_dists = 0.5 - np.abs(p)
        min_wall = np.min(wall_dists)

        # Pairwise distance
        # For N=50, calculating all pairs is cheap
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                dists[i, j] = d
                dists[j, i] = d
        min_pair = np.min(dists)

        r = min(min_wall, min_pair / 2.0)
        return r

    def soft_min_radius(p, beta):
        """
        Smooth surrogate for min(wall_dists, pair_dists/2) using log-sum-exp.
        We want to maximize this, so we minimize -soft_min.
        """
        # Wall distances
        wall_dists = 0.5 - np.abs(p)
        
        # Pair distances / 2
        pair_dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j]) / 2.0
                pair_dists[i, j] = d
                pair_dists[j, i] = d
        
        # Combine all "radius-like" values
        # We want soft-min of all wall_dists and all pair_dists/2
        all_vals = np.concatenate([wall_dists.ravel(), pair_dists.ravel()])
        
        # Log-sum-exp trick for numerical stability
        max_val = np.max(all_vals)
        if max_val < -1000:
            return max_val
        shifted = all_vals - max_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta + max_val
        return log_sum

    def neg_soft_min_radius(p, beta):
        return -soft_min_radius(p, beta)

    def local_search_lbfgs(initial_pos, time_limit, beta_start=20.0, beta_end=100.0):
        p = initial_pos.copy()
        current_r = compute_radius(p)
        best_r_local = current_r
        best_p_local = p.copy()
        
        # Anneal beta from soft to sharp
        num_stages = 5
        betas = np.linspace(beta_start, beta_end, num_stages)
        
        for beta in betas:
            if time.time() > time_limit:
                break
                
            # Define objective and gradient for this beta
            def obj_and_grad(x):
                p2 = x.reshape(N, 3)
                # Compute soft min radius
                wall_dists = 0.5 - np.abs(p2)
                pair_dists = np.zeros((N, N))
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(p2[i] - p2[j]) / 2.0
                        pair_dists[i, j] = d
                        pair_dists[j, i] = d
                        
                all_vals = np.concatenate([wall_dists.ravel(), pair_dists.ravel()])
                max_val = np.max(all_vals)
                shifted = all_vals - max_val
                exp_vals = np.exp(-beta * shifted)
                sum_exp = np.sum(exp_vals)
                
                # Soft min value
                soft_min_val = max_val - np.log(sum_exp) / beta
                
                # Gradient of soft min
                # d(soft_min)/dx = sum_i (exp(-beta*(v_i - max)) / sum_exp) * (-beta * dv_i/dx)
                # = -beta * sum_i (w_i * dv_i/dx) where w_i = exp(-beta*(v_i - max)) / sum_exp
                
                grad = np.zeros_like(p2)
                
                # Wall gradient
                wall_exp = exp_vals[:N*3]
                wall_w = wall_exp / sum_exp
                for i in range(N):
                    for k in range(3):
                        val = p2[i, k]
                        sign = 1.0 if val > 0 else -1.0
                        # d(wall_dist)/dx = -sign
                        # d(soft_min)/dx += -beta * w * (-sign) = beta * w * sign
                        grad[i, k] += beta * wall_w[i*3+k] * sign
                
                # Pair gradient
                pair_exp = exp_vals[N*3:]
                pair_w = pair_exp / sum_exp
                for i in range(N):
                    for j in range(i+1, N):
                        w = pair_w[(i*N + j)]
                        d_vec = p2[i] - p2[j]
                        dist = np.linalg.norm(d_vec)
                        if dist > 1e-12:
                            unit = d_vec / dist
                            # d(pair_dist/2)/dx_i = 0.5 * unit
                            # d(soft_min)/dx_i += -beta * w * (0.5 * unit)
                            grad[i] -= beta * w * 0.5 * unit
                            # d(pair_dist/2)/dx_j = -0.5 * unit
                            # d(soft_min)/dx_j += -beta * w * (-0.5 * unit) = beta * w * 0.5 * unit
                            grad[j] += beta * w * 0.5 * unit
                
                # We minimize -soft_min, so gradient is -grad_soft_min
                return -soft_min_val, -grad.reshape(-1)
            
            # Use L-BFGS-B
            res = minimize(
                obj_and_grad,
                p.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'maxfun': 10000, 'gtol': 1e-6}
            )
            
            new_p = res.x.reshape(N, 3)
            new_r = compute_radius(new_p)
            
            if new_r > best_r_local:
                best_r_local = new_r
                best_p_local = new_p.copy()
                p = new_p.copy()
            
            # Check time
            if time.time() > time_limit:
                break
        
        # Final polish: high-precision local optimization on active constraints
        p = best_p_local.copy()
        for _ in range(50):
            if time.time() > time_limit:
                break
            
            # Find active constraints
            min_pair_dist = 1.0
            min_pair_idx = (-1, -1)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
                        min_pair_idx = (i, j)
                        
            min_wall_dist = 1.0
            min_wall_idx = (-1, -1)
            for i in range(N):
                for k in range(3):
                    w = 0.5 - abs(p[i, k])
                    if w < min_wall_dist:
                        min_wall_dist = w
                        min_wall_idx = (i, k)
            
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # Move spheres in min pair apart
            if min_pair_idx[0] != -1:
                i, j = min_pair_idx
                d_vec = p[i] - p[j]
                dist = np.linalg.norm(d_vec)
                if dist > 1e-6:
                    d_vec = d_vec / dist
                    eps = 0.0005
                    p[i] += eps * d_vec
                    p[j] -= eps * d_vec
                    p = np.clip(p, -0.5, 0.5)
            
            # Move sphere in min wall inward
            if min_wall_idx[0] != -1:
                i, k = min_wall_idx
                sign = 1.0 if p[i, k] > 0 else -1.0
                p[i, k] -= 0.0005 * sign
                p = np.clip(p, -0.5, 0.5)
            
            new_r = compute_radius(p)
            if new_r > best_r_local:
                best_r_local = new_r
                best_p_local = p.copy()
        
        return best_p_local, best_r_local

    # Run local search from grid
    time_left = deadline - time.time()
    if time_left > 5:
        best_pos, best_r = local_search_lbfgs(pos, time.time() + time_left * 0.4)

    # If time remains, try a random restart
    time_left = deadline - time.time()
    if time_left > 10:
        rng = np.random.RandomState(42)
        rand_pos = rng.uniform(-0.4, 0.4, size=(N, 3))
        rand_pos, rand_r = local_search_lbfgs(rand_pos, time.time() + time_left * 0.8)
        
        if rand_r > best_r:
            best_r = rand_r
            best_pos = rand_pos

    # Final check
    final_r = compute_radius(best_pos)
    if final_r < 0:
        best_pos = grid[:N].copy()
        final_r = compute_radius(best_pos)
        
    return best_pos
