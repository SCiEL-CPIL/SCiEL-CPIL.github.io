"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        if not np.all(np.isfinite(c)):
            return -np.inf
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return -np.inf
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Distance to walls
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise: compute all N×N distances, extract upper triangle
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        d_pairs = d_full[tri_i, tri_j] / 2.0

        # Wall distances
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min value
        neg_beta_g = -beta * g_all
        log_sum_exp = logsumexp(neg_beta_g)
        val = -log_sum_exp / beta

        # Weights
        weights = np.exp(neg_beta_g - log_sum_exp)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        # Pairwise gradient
        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients (vectorized)
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws+N]
        grad_c[:, 0] += weights[ws+N:ws+2*N]
        grad_c[:, 1] -= weights[ws+2*N:ws+3*N]
        grad_c[:, 1] += weights[ws+3*N:ws+4*N]
        grad_c[:, 2] -= weights[ws+4*N:ws+5*N]
        grad_c[:, 2] += weights[ws+5*N:ws+6*N]

        return -val, -grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter):
        """Run L-BFGS-B on the smooth surrogate."""
        x0 = c_init.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)

        def fun(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return val

        def jac(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100,
                         'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with vectorized gradient.
        """
        # Refined beta schedule: start lower, go higher, more steps
        betas = [10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000, 400000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for idx, beta in enumerate(betas):
            if time.time() > deadline - 5:
                break

            remaining = deadline - time.time() - 5
            n_remaining = len(betas) - idx
            stage_time = max(1.0, remaining / n_remaining * 0.9)
            
            # More iterations for lower beta to explore, fewer for high beta to polish
            if beta < 100:
                max_iter = int(min(300, max(100, stage_time * 30)))
            elif beta < 1000:
                max_iter = int(min(200, max(50, stage_time * 20)))
            else:
                max_iter = int(min(100, max(20, stage_time * 10)))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        return c_best, r_best

    def generate_initializations():
        """
        Generate diverse initial configurations.
        """
        inits = []
        
        # 1. BCC Lattice (Body-Centered Cubic)
        # A BCC lattice has high packing density.
        # We generate a grid of BCC unit cells and select 50 points.
        n_cell = 4
        cell_size = 1.0 / n_cell
        bcc_points = []
        for iz in range(n_cell):
            for iy in range(n_cell):
                for ix in range(n_cell):
                    cx = -0.5 + cell_size * (ix + 0.5)
                    cy = -0.5 + cell_size * (iy + 0.5)
                    cz = -0.5 + cell_size * (iz + 0.5)
                    bcc_points.append([cx, cy, cz])
                    bcc_points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
        bcc_arr = np.array(bcc_points)
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        if len(bcc_arr) >= N:
            # Shuffle to pick a random subset, or just take first N? 
            # Taking first N might be biased. Let's shuffle.
            np.random.seed(42)
            perm = np.random.permutation(len(bcc_arr))
            inits.append(bcc_arr[perm[:N]].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            inits.append(np.vstack([bcc_arr, extra]))

        # 2. FCC Lattice (Face-Centered Cubic)
        # FCC has even higher density in infinite space.
        fcc_points = []
        for iz in range(n_cell):
            for iy in range(n_cell):
                for ix in range(n_cell):
                    cx = -0.5 + cell_size * (ix + 0.5)
                    cy = -0.5 + cell_size * (iy + 0.5)
                    cz = -0.5 + cell_size * (iz + 0.5)
                    fcc_points.append([cx, cy, cz])
                    # Face centers
                    fcc_points.append([cx + cell_size/2, cy + cell_size/2, cz])
                    fcc_points.append([cx + cell_size/2, cy, cz + cell_size/2])
                    fcc_points.append([cx, cy + cell_size/2, cz + cell_size/2])
        fcc_arr = np.array(fcc_points)
        mask = np.all(np.abs(fcc_arr) <= 0.5, axis=1)
        fcc_arr = fcc_arr[mask]
        if len(fcc_arr) >= N:
            np.random.seed(43)
            perm = np.random.permutation(len(fcc_arr))
            inits.append(fcc_arr[perm[:N]].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(fcc_arr), 3))
            inits.append(np.vstack([fcc_arr, extra]))

        # 3. Jittered Grid
        # Start with a grid and add random jitter to break symmetry.
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        np.random.seed(44)
        jitter = np.random.uniform(-0.05, 0.05, size=(N, 3))
        jittered = grid + jitter
        # Clamp to bounds
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)

        # 4. Random with Min Distance (Poisson Disk-like)
        # Try to place points with a minimum separation.
        np.random.seed(45)
        pts = []
        min_dist = 0.15 # Initial guess for min distance
        for _ in range(N):
            for _ in range(100): # Try 100 times to place a point
                p = np.random.uniform(-0.49, 0.49, size=3)
                if len(pts) == 0:
                    pts.append(p)
                    break
                dists = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists) > min_dist:
                    pts.append(p)
                    break
        if len(pts) < N:
            # Fill remaining with random
            while len(pts) < N:
                pts.append(np.random.uniform(-0.49, 0.49, size=3))
        inits.append(np.array(pts))

        # 5. Layered Initialization (HCP-like)
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c_layered = np.zeros((N, 3))
        pt_idx = 0
        for layer in range(n_layers):
            z = -0.4 + layer * layer_height
            n_pts = points_per_layer + (1 if layer < remaining_pts else 0)
            n_x = int(np.ceil(np.sqrt(n_pts)))
            n_y = int(np.ceil(n_pts / n_x))
            spacing_x = 0.7 / (n_x + 0.5)
            spacing_y = 0.7 / (n_y + 0.5)
            for ix in range(n_x):
                for iy in range(n_y):
                    if pt_idx >= N:
                        break
                    x = -0.35 + ix * spacing_x
                    y = -0.35 + iy * spacing_y
                    if (ix % 2 == 1):
                        y += spacing_y / 2
                    c_layered[pt_idx] = [x, y, z]
                    pt_idx += 1
            if pt_idx >= N:
                break
        inits.append(c_layered)

        return inits

    # Generate multiple initial configurations
    initial_configs = generate_initializations()

    best_centers = None
    best_r = -np.inf

    # Try each initial configuration
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break

        # Run annealed smooth optimization
        c_opt, r_opt = annealed_optimization(init)

        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()

        # Final high-precision polish with very high beta
        if time.time() < deadline - 5:
            beta_final = 500000
            c_polished = run_smooth_optimization(best_centers, beta_final, 200)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = c_polished.copy()

    # If no valid result, fall back to simple grid
    if best_centers is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    # Final sanity check and clamp
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
