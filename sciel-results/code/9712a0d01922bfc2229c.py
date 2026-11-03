"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
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

        # Pairwise gradient: for each pair (i,j) with weight w_k,
        # grad_c[i] += w_k * 0.5 * (c_i - c_j) / d_ij
        # grad_c[j] -= w_k * 0.5 * (c_i - c_j) / d_ij
        # Vectorize: diff_ij = c[tri_i] - c[tri_j], d_ij = d_full[tri_i, tri_j]
        diff_ij = c[tri_i] - c[tri_j]  # (n_pairs, 3)
        d_ij = d_full[tri_i, tri_j]    # (n_pairs,)
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]  # (n_pairs, 3)
        w_pair = weights[:n_pairs] * 0.5 / safe_d   # (n_pairs,)
        contrib = w_pair[:, np.newaxis] * unit_vecs  # (n_pairs, 3)

        # Accumulate using np.add.at for unbuffered scatter-add
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
        bounds = [(-0.499, 0.499)] * (N * 3)

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
        betas = [30, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 30720, 61440, 122880, 245760, 491520]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for idx, beta in enumerate(betas):
            if time.time() > deadline - 2:
                break

            remaining = deadline - 2 - time.time()
            n_remaining = len(betas) - idx
            # Allocate time proportionally, with more for early stages
            stage_time = max(1.5, remaining / n_remaining * 0.9)
            # Higher beta needs fewer iterations as the landscape becomes sharper
            max_iter = int(min(350, max(30, stage_time * 30 / (1 + 0.15 * np.log2(beta)))))

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

        # 1. Simple grid (4x4x4 = 64, take first 50)
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N].copy())

        # 2. BCC-inspired: body-centered cubic lattice
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
            inits.append(bcc_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            inits.append(np.vstack([bcc_arr, extra]))

        # 3. FCC-inspired: face-centered cubic lattice
        # FCC has 4 points per unit cell: corners + face centers
        n_cell_fcc = 3
        cell_size_fcc = 1.0 / n_cell_fcc
        fcc_points = []
        for iz in range(n_cell_fcc):
            for iy in range(n_cell_fcc):
                for ix in range(n_cell_fcc):
                    cx = -0.5 + cell_size_fcc * (ix + 0.5)
                    cy = -0.5 + cell_size_fcc * (iy + 0.5)
                    cz = -0.5 + cell_size_fcc * (iz + 0.5)
                    fcc_points.append([cx, cy, cz])
                    # Face centers
                    fcc_points.append([cx + cell_size_fcc/2, cy + cell_size_fcc/2, cz])
                    fcc_points.append([cx + cell_size_fcc/2, cy, cz + cell_size_fcc/2])
                    fcc_points.append([cx, cy + cell_size_fcc/2, cz + cell_size_fcc/2])
        fcc_arr = np.array(fcc_points)
        mask_fcc = np.all(np.abs(fcc_arr) <= 0.5, axis=1)
        fcc_arr = fcc_arr[mask_fcc]
        if len(fcc_arr) >= N:
            inits.append(fcc_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(fcc_arr), 3))
            inits.append(np.vstack([fcc_arr, extra]))

        # 4. Random rotations of grid
        np.random.seed(42)
        for trial in range(4):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
            Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
            R = Ry @ Rx
            base = np.random.uniform(-0.3, 0.3, size=(N, 3))
            rotated = base @ R.T
            max_abs = np.max(np.abs(rotated))
            if max_abs > 0:
                rotated = rotated * (0.45 / max_abs)
            inits.append(rotated)

        # 5. Layered initialization
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c_layered = np.zeros((N, 3))
        idx = 0
        for layer in range(n_layers):
            n_in_layer = points_per_layer + (1 if layer < remaining_pts else 0)
            z_coord = -0.4 + layer_height * (layer + 0.5)
            m_layer = int(np.ceil(np.sqrt(n_in_layer)))
            spacing = 0.8 / m_layer
            for i in range(n_in_layer):
                ix = i % m_layer
                iy = i // m_layer
                x = -0.4 + spacing * (ix + 0.5)
                y = -0.4 + spacing * (iy + 0.5)
                c_layered[idx] = [x, y, z_coord]
                idx += 1
        inits.append(c_layered)

        # 6. Random uniform
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

        return inits

    # Main optimization loop
    inits = generate_initializations()

    best_c = None
    best_r = -1.0

    # Reserve time for final polish
    polish_reserve = 10.0
    search_deadline = deadline - polish_reserve

    for c_init in inits:
        if time.time() > search_deadline - 3:
            break

        c_init = np.clip(c_init, -0.49, 0.49)
        c_opt, r_opt = annealed_optimization(c_init)

        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # Final high-precision polish with multiple beta stages
    if best_c is not None and time.time() < deadline - 3:
        polish_betas = [200000, 400000, 800000, 1600000]
        for beta in polish_betas:
            if time.time() > deadline - 2:
                break
            remaining = deadline - 2 - time.time()
            max_iter = int(min(300, max(50, remaining * 20)))
            c_polished = run_smooth_optimization(best_c, beta, max_iter)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()

    # Fallback
    if best_c is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    best_c = np.clip(best_c, -0.4999, 0.4999)

    final_r = get_radius(best_c)
    if final_r <= 0:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
