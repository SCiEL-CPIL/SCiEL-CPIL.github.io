"""
Sphere packing: 50 equal spheres in unit cube.

Strategy:
- Smooth surrogate optimization with beta annealing (log-sum-exp soft-min).
- Constraint-based refinement using active set identification.
- Multiple diverse initializations (FCC, BCC, HCP, grid, layered, random).
- Deadline guarding.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import lstsq

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 135.0


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3) or not np.all(np.isfinite(c)):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise distances
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
        lse = logsumexp(neg_beta_g)
        val = -lse / beta

        # Weights
        weights = np.exp(neg_beta_g - lse)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws + N]
        grad_c[:, 0] += weights[ws + N:ws + 2 * N]
        grad_c[:, 1] -= weights[ws + 2 * N:ws + 3 * N]
        grad_c[:, 1] += weights[ws + 3 * N:ws + 4 * N]
        grad_c[:, 2] -= weights[ws + 4 * N:ws + 5 * N]
        grad_c[:, 2] += weights[ws + 5 * N:ws + 6 * N]

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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def run_direct_optimization(c_init, max_iter):
        """Run L-BFGS-B on the exact non-smooth objective."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def fun(x):
            return -get_radius(x.reshape(N, 3))

        try:
            result = minimize(
                fun, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def active_set_refinement(c_init, r_target, max_iter=100):
        """
        Active-set refinement: identify constraints active at r_target and
        optimize only over those degrees of freedom.
        """
        c = c_init.copy()
        r = get_radius(c)
        if r >= r_target:
            return c, r

        # Identify active constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        # If the bottleneck is pair distance, try to increase it
        if min_pair_dist / 2.0 < min_wall_dist:
            # Find closest pairs
            pair_dists = d[tri_i, tri_j]
            close_pairs = np.where(pair_dists < min_pair_dist * 1.01)[0]
            if len(close_pairs) > 0:
                # Perturb points involved in close pairs
                involved = np.unique(np.concatenate([tri_i[close_pairs], tri_j[close_pairs]]))
                rng = np.random.RandomState(42)
                perturb = rng.normal(0, 0.005, size=(len(involved), 3))
                c[involved] += perturb
                c = np.clip(c, -0.499, 0.499)
                r_new = get_radius(c)
                if r_new > r:
                    c, r = c, r_new

        # Final direct optimization
        c_polished = run_direct_optimization(c, max_iter)
        r_polished = get_radius(c_polished)
        if r_polished > r:
            c, r = c_polished, r_polished

        return c, r

    def constraint_based_refinement(c_init, max_iter=200):
        """
        Constraint-based refinement: solve linear feasibility problem for active constraints.
        This can escape local optima that smooth surrogates cannot.
        """
        c = c_init.copy()
        r = get_radius(c)

        if remaining() < 5.0:
            return c, r

        # Get current radius
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        r_current = min(min_pair_dist / 2.0, min_wall_dist)

        # Target radius slightly above current
        r_target = r_current + 0.001

        # Build constraint matrix
        # Constraints: |c_i - c_j| >= 2*r_target for all pairs
        #               |c_i_k| <= 0.5 - r_target for all walls
        #
        # For pairs: (c_i - c_j)^T (c_i - c_j) >= (2*r_target)^2
        # For walls: |c_i_k| <= 0.5 - r_target

        # Use active constraints only (those close to being violated)
        pair_dists = d[tri_i, tri_j]
        active_pairs = np.where(pair_dists < 2 * r_target + 0.01)[0]

        # Wall constraints
        wall_constraints = []
        for i in range(N):
            for k in range(3):
                if abs(c[i, k]) > 0.5 - r_target - 0.01:
                    wall_constraints.append((i, k))

        if len(active_pairs) == 0 and len(wall_constraints) == 0:
            return c, r_current

        # Build A matrix and b vector for linear constraints
        # Pair constraint: ||c_i - c_j|| >= 2*r_target
        # Linearize: (c_i - c_j)^T * (c_i0 - c_j0) >= ||c_i0 - c_j0||^2 / 2
        # where c_i0, c_j0 are current positions

        A_rows = []
        b_rows = []

        for idx in active_pairs:
            i, j = tri_i[idx], tri_j[idx]
            d_ij = c[i] - c[j]
            norm_dij = np.linalg.norm(d_ij)
            if norm_dij > 1e-10:
                # Constraint: (c_i - c_j)^T d_ij >= ||d_ij||^2
                # This ensures c_i moves away from c_j along d_ij direction
                a_row = np.zeros(3 * N)
                a_row[i * 3:(i + 1) * 3] = d_ij / norm_dij
                a_row[j * 3:(j + 1) * 3] = -d_ij / norm_dij
                b_val = norm_dij - 2 * r_target + 0.005  # slight margin
                A_rows.append(a_row)
                b_rows.append(b_val)

        for i, k in wall_constraints:
            a_row = np.zeros(3 * N)
            if c[i, k] > 0:
                a_row[i * 3 + k] = 1.0
                b_val = 0.5 - r_target
            else:
                a_row[i * 3 + k] = -1.0
                b_val = -(0.5 - r_target)
            A_rows.append(a_row)
            b_rows.append(b_val)

        if len(A_rows) == 0:
            return c, r_current

        A = np.array(A_rows)
        b = np.array(b_rows)

        # Solve least squares problem: minimize ||Ax - b||^2 subject to bounds
        # This finds x that best satisfies all active constraints
        x0 = c.flatten()
        bounds = [(-0.499, 0.499)] * (3 * N)

        def fun_constraint(x):
            # Penalize constraint violations
            Ax = A @ x
            violations = np.maximum(0, b - Ax)
            return np.sum(violations ** 2)

        try:
            result = minimize(
                fun_constraint, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14}
            )
            c_new = np.clip(result.x.reshape(N, 3), -0.499, 0.499)
            r_new = get_radius(c_new)
            if r_new > r_current:
                c = c_new
                r = r_new
        except Exception:
            pass

        # Polish with direct optimization
        if remaining() > 10.0:
            c_polished = run_direct_optimization(c, max_iter=100)
            r_polished = get_radius(c_polished)
            if r_polished > r:
                c, r = c_polished, r_polished

        return c, r

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with aggressive beta schedule and time-proportional
        iteration allocation.
        """
        betas = [10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000, 400000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for idx, beta in enumerate(betas):
            if remaining() < 3.0:
                break

            n_remaining = len(betas) - idx
            stage_time = max(1.0, remaining() / n_remaining * 0.85)
            # Higher beta needs fewer iterations
            max_iter = int(min(200, max(20, stage_time * 15 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        # Active-set refinement
        if remaining() > 5.0:
            c_refined, r_refined = active_set_refinement(c_best, r_best)
            if r_refined > r_best:
                c_best = c_refined.copy()
                r_best = r_refined

        # Constraint-based refinement (new)
        if remaining() > 15.0:
            c_constrained, r_constrained = constraint_based_refinement(c_best)
            if r_constrained > r_best:
                c_best = c_constrained.copy()
                r_best = r_constrained

        # Final direct polish
        if remaining() > 3.0:
            c_polished = run_direct_optimization(c_best, max_iter=150)
            r_polished = get_radius(c_polished)
            if r_polished > r_best:
                c_best = c_polished.copy()
                r_best = r_polished

        return c_best, r_best

    def make_fcc_dense_init():
        """
        Dense FCC lattice initialization.
        FCC has points at corners and face centers of cubic cells.
        """
        a = 0.25  # lattice parameter
        positions = []
        # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        basis = np.array([[0, 0, 0],
                          [0.5, 0.5, 0],
                          [0.5, 0, 0.5],
                          [0, 0.5, 0.5]])

        n_cells = 4
        for iz in range(n_cells):
            for iy in range(n_cells):
                for ix in range(n_cells):
                    origin = np.array([ix, iy, iz]) * a
                    for b in basis:
                        pos = origin + b * a
                        if np.all(np.abs(pos - 0.5) <= 0.48):  # within cube
                            positions.append(pos)

        positions = np.array(positions, dtype=np.float64)

        if len(positions) >= N:
            # Select N points with best local spacing
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        else:
            # Add random points to fill up
            rng = np.random.RandomState(123)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return np.clip(positions[:N].copy(), -0.499, 0.499)

    def make_grid_init():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_init():
        """BCC-inspired: body-centered cubic lattice."""
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
                    bcc_points.append([cx + cell_size / 2, cy + cell_size / 2, cz + cell_size / 2])
        bcc_arr = np.array(bcc_points)
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        if len(bcc_arr) >= N:
            return bcc_arr[:N].copy()
        else:
            extra = np.random.RandomState(7).uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            return np.vstack([bcc_arr, extra])

    def make_halton_init():
        """
        Halton sequence initialization for low-discrepancy spatial distribution.
        This acts as a 'deterministic chaotic-like' spread, avoiding lattice traps.
        """
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((N, 3))
        for i in range(N):
            points[i, 0] = halton(i + 1, 2)
            points[i, 1] = halton(i + 1, 3)
            points[i, 2] = halton(i + 1, 5)
        
        # Map from [0, 1]^3 to [-0.49, 0.49]^3 to keep safely inside
        points = points * 0.98 - 0.49
        return np.clip(points, -0.499, 0.499)

    def make_rotated_random_init(seed):
        """Random points rotated by a random orthogonal matrix."""
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        R = Ry @ Rx
        base = rng.uniform(-0.35, 0.35, size=(N, 3))
        rotated = base @ R.T
        max_abs = np.max(np.abs(rotated))
        if max_abs > 0:
            rotated = rotated * (0.45 / max_abs)
        return np.clip(rotated, -0.499, 0.499)

    def make_layered_init():
        """Layered initialization: distribute points across 5 layers."""
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c = np.zeros((N, 3))
        idx = 0
        for layer in range(n_layers):
            z = -0.4 + layer_height * (layer + 0.5)
            n_in_layer = points_per_layer + (1 if layer < remaining_pts else 0)
            # Distribute points in a grid within the layer
            n_x = int(np.ceil(np.sqrt(n_in_layer)))
            n_y = n_in_layer // n_x + (1 if n_in_layer % n_x else 0)
            x_spacing = 0.8 / max(n_x, 1)
            y_spacing = 0.8 / max(n_y, 1)
            count = 0
            for iy in range(n_y):
                for ix in range(n_x):
                    if count >= n_in_layer:
                        break
                    x = -0.4 + x_spacing * (ix + 0.5)
                    y = -0.4 + y_spacing * (iy + 0.5)
                    c[idx] = [x, y, z]
                    idx += 1
                    count += 1
        return np.clip(c, -0.499, 0.499)

    def make_hcp_init():
        """HCP-inspired initialization."""
        a = 0.3
        c = 0.5  # c/a ratio for ideal HCP
        positions = []
        # HCP basis in hexagonal coordinates
        basis = np.array([[0, 0, 0],
                          [1/3, 2/3, 0.5]])

        n_cells = 4
        for iz in range(n_cells):
            for iy in range(n_cells):
                for ix in range(n_cells):
                    # Hexagonal to Cartesian
                    x = a * (ix + iy / 2)
                    y = a * (iy * np.sqrt(3) / 2)
                    z = c * iz
                    for b in basis:
                        pos = np.array([x + a * b[0], y + a * b[1], z + c * b[2]])
                        if np.all(np.abs(pos - 0.5) <= 0.48):
                            positions.append(pos)

        positions = np.array(positions, dtype=np.float64)

        if len(positions) >= N:
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        else:
            rng = np.random.RandomState(456)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return np.clip(positions[:N].copy(), -0.499, 0.499)

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = get_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    # Define initializations
    initializations = [
        ("fcc_dense", make_fcc_dense_init),
        ("grid", make_grid_init),
        ("bcc", make_bcc_init),
        ("hcp", make_hcp_init),
        ("layered", make_layered_init),
        ("halton", make_halton_init),
        ("rot0", lambda: make_rotated_random_init(0)),
        ("rot1", lambda: make_rotated_random_init(42)),
        ("rot2", lambda: make_rotated_random_init(123)),
        ("rot3", lambda: make_rotated_random_init(777)),
    ]

    # Stage 1: Try each initialization with aggressive annealing
    for name, init_func in initializations:
        if remaining() < 10:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499, 0.499)

            pos_opt, r_opt = annealed_optimization(positions)
            update_best(pos_opt)

        except Exception:
            continue

    # Phase 2: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 20:
        n_perturb = 6
        for k in range(n_perturb):
            if remaining() < 6:
                break
            rng = np.random.RandomState(2000 + k)
            noise_scale = 0.010 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499, 0.499)

            pos_opt, r_opt = annealed_optimization(perturbed)
            update_best(pos_opt)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499, 0.499)
    else:
        best_positions = make_grid_init()
        best_positions = np.clip(best_positions, -0.499, 0.499)

    # Final high-precision polish if time remains
    if remaining() > 6:
        try:
            polished = run_smooth_optimization(best_positions, beta=400000, max_iter=200)
            polished = run_direct_optimization(polished, max_iter=150)
            if get_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499, 0.499)
        except Exception:
            pass

    return best_positions
