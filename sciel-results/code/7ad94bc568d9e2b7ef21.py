"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s limit

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
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100,
                         'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def active_set_polish(c_init, time_budget):
        """
        Identify binding constraints and polish with a focused smooth optimization.
        """
        c = c_init.copy()
        r = get_radius(c)

        for _ in range(4):
            if time.time() > time_budget:
                break

            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair_dist = np.min(d)

            dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            min_wall_dist = np.min(dist_to_walls)

            r_current = min(min_pair_dist / 2.0, min_wall_dist)

            tol = 0.005 * r_current + 1e-9

            pair_dists = d[tri_i, tri_j] / 2.0
            binding_pairs = np.where(pair_dists <= r_current + tol)[0]

            dist_pos = 0.5 - c
            dist_neg = c + 0.5

            pos_flat = dist_pos.flatten()
            neg_flat = dist_neg.flatten()

            pos_bind_flat = np.where(pos_flat <= r_current + tol)[0]
            neg_bind_flat = np.where(neg_flat <= r_current + tol)[0]

            pos_bind_i = pos_bind_flat // 3
            pos_bind_k = pos_bind_flat % 3
            neg_bind_i = neg_bind_flat // 3
            neg_bind_k = neg_bind_flat % 3

            n_bind_pos = len(pos_bind_i)
            n_bind_neg = len(neg_bind_i)
            n_bind_pairs = len(binding_pairs)

            if n_bind_pairs == 0 and n_bind_pos == 0 and n_bind_neg == 0:
                return c, r

            def binding_objective_and_grad(c_flat, beta):
                c = c_flat.reshape(N, 3)

                if np.any(np.abs(c) > 0.5 + 1e-9):
                    return 1e6, np.zeros_like(c_flat)

                if n_bind_pairs > 0:
                    diff_bind = c[tri_i[binding_pairs]] - c[tri_j[binding_pairs]]
                    d_bind = np.linalg.norm(diff_bind, axis=1) / 2.0
                else:
                    d_bind = np.array([])
                    diff_bind = np.zeros((0, 3))

                if n_bind_pos > 0:
                    wall_pos = 0.5 - c[pos_bind_i, pos_bind_k]
                else:
                    wall_pos = np.array([])

                if n_bind_neg > 0:
                    wall_neg = c[neg_bind_i, neg_bind_k] + 0.5
                else:
                    wall_neg = np.array([])

                wall_dists = np.concatenate([wall_pos, wall_neg])
                g_all = np.concatenate([d_bind, wall_dists])

                if len(g_all) == 0:
                    return 0.0, np.zeros_like(c_flat)

                neg_beta_g = -beta * g_all
                log_sum_exp = logsumexp(neg_beta_g)
                val = -log_sum_exp / beta

                weights = np.exp(neg_beta_g - log_sum_exp)

                grad_c = np.zeros_like(c)

                if n_bind_pairs > 0:
                    safe_d = np.where(d_bind * 2 > 1e-14, d_bind * 2, 1.0)
                    unit_vecs = diff_bind / safe_d[:, np.newaxis]
                    w_pair = weights[:n_bind_pairs] * 0.5 / safe_d
                    contrib = w_pair[:, np.newaxis] * unit_vecs
                    np.add.at(grad_c, tri_i[binding_pairs], contrib)
                    np.add.at(grad_c, tri_j[binding_pairs], -contrib)

                if n_bind_pos > 0:
                    w_pos = weights[n_bind_pairs:n_bind_pairs + n_bind_pos]
                    np.add.at(grad_c, (pos_bind_i, pos_bind_k), w_pos[:, np.newaxis])

                if n_bind_neg > 0:
                    w_neg = weights[n_bind_pairs + n_bind_pos:n_bind_pairs + n_bind_pos + n_bind_neg]
                    np.add.at(grad_c, (neg_bind_i, neg_bind_k), -w_neg[:, np.newaxis])

                return -val, -grad_c.flatten()

            x0 = c.flatten()
            bounds = [(-0.4999, 0.4999)] * (N * 3)

            def fun(x):
                val, grad = binding_objective_and_grad(x, 1000000)
                return val

            def jac(x):
                val, grad = binding_objective_and_grad(x, 1000000)
                return grad

            try:
                result = minimize(
                    fun, x0, jac=jac,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 50,
                             'maxfun': 3000, 'maxcor': 40}
                )
                c_new = result.x.reshape(N, 3)
                r_new = get_radius(c_new)
                if r_new > r:
                    c = c_new
                    r = r_new
                else:
                    break
            except Exception:
                pass

        return c, r

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with vectorized gradient.
        """
        betas = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for beta in betas:
            if time.time() > deadline:
                break

            remaining = deadline - time.time()
            idx = betas.index(beta)
            n_remaining = len(betas) - idx
            stage_time = max(1.0, remaining / n_remaining * 0.8)
            max_iter = int(min(400, max(20, stage_time * 25 / (1 + 0.1 * np.log2(max(beta, 1))))))

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

        # 1b. Offset grid
        ticks_shifted = (np.arange(m) + 0.25) / m - 0.5
        grid_shifted = np.array([(x, y, z) for z in ticks for y in ticks_shifted for x in ticks])
        inits.append(grid_shifted[:N].copy())

        # 2. BCC-inspired
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

        # 3. FCC-inspired
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
                    fcc_points.append([cx + cell_size_fcc/2, cy, cz])
                    fcc_points.append([cx, cy + cell_size_fcc/2, cz])
                    fcc_points.append([cx, cy, cz + cell_size_fcc/2])
        fcc_arr = np.array(fcc_points)
        mask_fcc = np.all(np.abs(fcc_arr) <= 0.5, axis=1)
        fcc_arr = fcc_arr[mask_fcc]
        if len(fcc_arr) >= N:
            inits.append(fcc_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(fcc_arr), 3))
            inits.append(np.vstack([fcc_arr, extra]))

        # 4. HCP-inspired (hexagonal close packing)
        n_layers_hcp = 5
        a = 0.35  # in-plane spacing
        c_param = 1.633 * a  # c/a ratio for HCP
        hcp_points = []
        for layer in range(n_layers_hcp):
            z = -0.5 + c_param * (layer + 0.5)
            if z > 0.5:
                break
            # Hexagonal lattice in xy
            n_x = 5
            n_y = 5
            for iy in range(n_y):
                for ix in range(n_x):
                    x = -0.5 + a * (ix + 0.5 * (iy % 2))
                    y = -0.5 + a * np.sqrt(3)/2 * (iy + 0.5)
                    if abs(x) <= 0.5 and abs(y) <= 0.5:
                        hcp_points.append([x, y, z])
        hcp_arr = np.array(hcp_points)
        mask_hcp = np.all(np.abs(hcp_arr) <= 0.5, axis=1)
        hcp_arr = hcp_arr[mask_hcp]
        if len(hcp_arr) >= N:
            inits.append(hcp_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(hcp_arr), 3))
            inits.append(np.vstack([hcp_arr, extra]))

        # 5. Random rotations of grid
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

        # 6. Layered initialization
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

        # 7. Random uniform
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

        # 8. Halton sequence (low discrepancy)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        halton_pts = np.zeros((N, 3))
        for i in range(N):
            idx = i + 1
            halton_pts[i, 0] = halton(idx, 2) - 0.5
            halton_pts[i, 1] = halton(idx, 3) - 0.5
            halton_pts[i, 2] = halton(idx, 5) - 0.5
        halton_pts *= 0.9
        inits.append(halton_pts)

        # 9. Fibonacci sphere-like distribution projected to cube
        golden = (1 + np.sqrt(5)) / 2
        fib_pts = np.zeros((N, 3))
        for i in range(N):
            t = i / N
            phi = np.arccos(1 - 2 * t)
            theta = 2 * np.pi * i / golden
            fib_pts[i, 0] = 0.45 * np.sin(phi) * np.cos(theta)
            fib_pts[i, 1] = 0.45 * np.sin(phi) * np.sin(theta)
            fib_pts[i, 2] = 0.45 * np.cos(phi)
        inits.append(fib_pts)

        return inits

    # Main optimization loop
    inits = generate_initializations()

    best_c = None
    best_r = -1.0

    for c_init in inits:
        if time.time() > deadline - 15:
            break

        c_init = np.clip(c_init, -0.49, 0.49)
        c_opt, r_opt = annealed_optimization(c_init)

        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 10:
        for beta in [500000, 1000000, 2000000]:
            if time.time() > deadline - 6:
                break
            remaining = deadline - 6 - time.time()
            max_iter = int(min(400, max(50, remaining * 15)))
            c_polished = run_smooth_optimization(best_c, beta, max_iter)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()

        # Active-set polish
        if time.time() < deadline - 3:
            c_polished, r_polished = active_set_polish(best_c, deadline - time.time() - 2)
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
