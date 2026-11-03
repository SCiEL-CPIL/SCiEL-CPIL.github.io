"""Combined strategy: Vectorized smooth-surrogate (log-sum-exp) with analytic gradients,
beta-annealing, Nesterov-accelerated active-set gradient ascent on the exact min-distance,
plus a short global search and final polish. For 50 equal spheres in [-1/2,1/2]^3."""
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
        """Exact minimum radius for positions (N,3) in [-1/2,1/2]^3."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        diff = positions[:, None, :] - positions[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        min_wall = np.min(0.5 - np.abs(positions))
        return float(min(min_pair, min_wall))

    # Pre-compute triangle indices for pairwise terms
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Smooth surrogate: -soft_min of all radius-candidates (pairwise half-distances + wall distances).
        Returns (value, gradient) for minimization.
        """
        c = c_flat.reshape(N, 3)

        # Pairwise half-distances
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0

        # Wall distances: 6 per point
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min via log-sum-exp
        scaled = -beta * g_all
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        val = -lse / beta

        # Weights for gradient
        weights = np.exp(scaled - lse)

        grad_c = np.zeros_like(c)

        # Pairwise gradient contribution
        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        weighted_diffs = factors[:, None] * diff
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)

        # Wall gradient contribution
        w0 = n_pairs
        grad_c[:, 0] -= weights[w0: w0 + N]
        grad_c[:, 0] += weights[w0 + N: w0 + 2 * N]
        grad_c[:, 1] -= weights[w0 + 2 * N: w0 + 3 * N]
        grad_c[:, 1] += weights[w0 + 3 * N: w0 + 4 * N]
        grad_c[:, 2] -= weights[w0 + 4 * N: w0 + 5 * N]
        grad_c[:, 2] += weights[w0 + 5 * N: w0 + 6 * N]

        return -val, -grad_c.flatten()

    def nesterov_active_set_gradient_ascent(c_init, maxiter=500, lr_start=0.002, lr_end=0.0001):
        """
        Nesterov-accelerated gradient ascent on the exact min-distance objective.
        Uses active-set (k-NN) subgradient approximation for efficiency.
        """
        if remaining() < 3:
            return c_init

        c = c_init.copy()
        v = np.zeros_like(c)
        best_c = c.copy()
        best_r = compute_radius(c)

        for it in range(maxiter):
            if remaining() < 2:
                break

            lr = lr_start * (lr_end / lr_start) ** (it / max(1, maxiter))

            # Compute active-set subgradient
            # Pairwise: find closest pairs
            diff = c[:, None, :] - c[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)

            # Wall distances
            wall_d = np.min(0.5 - np.abs(c), axis=1)

            current_min = min(np.min(d) / 2.0, np.min(wall_d))

            # Active set: pairs within 1.2 * current_min and walls within 1.2 * current_min
            threshold = max(1.2 * current_min, 0.01)

            # Pairwise active set
            active_pair_mask = (d < 2.0 * threshold) & ~np.eye(N, dtype=bool)
            active_pair_mask = active_pair_mask & (np.triu(np.ones((N, N), dtype=bool), 1))
            ai, aj = np.where(active_pair_mask)

            grad = np.zeros_like(c)
            if len(ai) > 0:
                d_active = d[ai, aj]
                d_safe = np.maximum(d_active, 1e-12)
                unit_diffs = (c[ai] - c[aj]) / d_safe[:, None]
                # Subgradient: push apart
                np.add.at(grad, ai, unit_diffs)
                np.add.at(grad, aj, -unit_diffs)

            # Wall active set
            for i in range(N):
                for dim in range(3):
                    if c[i, dim] > 0:
                        if 0.5 - c[i, dim] < threshold:
                            grad[i, dim] -= 1.0
                    else:
                        if c[i, dim] + 0.5 < threshold:
                            grad[i, dim] += 1.0

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-15:
                grad = grad / grad_norm

            # Nesterov update
            v = 0.9 * v + lr * grad
            c_new = c + v
            c_new = np.clip(c_new, -0.499999, 0.499999)

            # Check improvement
            r_new = compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            c = c_new

        return best_c

    def local_optimize_smooth(c_init, beta_start=5.0, beta_end=80.0, n_beta=5, maxiter_per=100):
        """L-BFGS-B on smooth surrogate with beta annealing."""
        if remaining() < 5:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)
        betas = np.linspace(beta_start, beta_end, n_beta)
        pos = c_init.flatten()

        for beta in betas:
            if remaining() < 3:
                break

            def obj_flat(x):
                val, _ = smooth_objective_and_grad(x, beta)
                return val

            def grad_flat(x):
                _, grad = smooth_objective_and_grad(x, beta)
                return grad

            try:
                res = opt.minimize(obj_flat, pos, jac=grad_flat, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': maxiter_per, 'ftol': 1e-15, 'gtol': 1e-10})
                pos = res.x
            except Exception:
                pass

        return np.clip(pos.reshape(N, 3), -0.499999, 0.499999)

    def local_optimize_direct(c_init, maxiter=200):
        """L-BFGS-B on the exact nonsmooth objective."""
        if remaining() < 5:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(obj_flat, c_init.flatten(), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def make_initial_grid():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        positions = []
        a = 0.5
        spacing = (2 * a) / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * spacing
                    y = -a + (iy + 0.5) * spacing
                    z = -a + (iz + 0.5) * spacing
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + spacing / 2, y + spacing / 2, z + spacing / 2])
        return np.array(positions[:N])

    def make_fcc_like():
        a = 0.35
        positions = []
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    x = (ix - 1.5) * a
                    y = (iy - 1.5) * a
                    z = (iz - 1.5) * a
                    positions.append([x, y, z])
                    if ix < 3:
                        positions.append([x + a / 2, y, z])
                    if iy < 3:
                        positions.append([x, y + a / 2, z])
                    if iz < 3:
                        positions.append([x, y, z + a / 2])
                    if ix < 3 and iy < 3:
                        positions.append([x + a / 2, y + a / 2, z])
                    if ix < 3 and iz < 3:
                        positions.append([x + a / 2, y, z + a / 2])
                    if iy < 3 and iz < 3:
                        positions.append([x, y + a / 2, z + a / 2])

        positions = np.array(positions, dtype=np.float64)
        mask = np.all(np.abs(positions) <= 0.48, axis=1)
        positions = positions[mask]

        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        elif len(positions) < N:
            rng = np.random.RandomState(0)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return positions[:N].copy()

    def make_random(seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    def coordinate_descent(positions, maxiter=50):
        if remaining() < 2:
            return positions
        pos = positions.copy()
        step = 0.005
        for it in range(maxiter):
            if remaining() < 1.5:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    old = pos[i, dim]
                    r_cur = compute_radius(pos)
                    pos[i, dim] = min(0.499999, old + step)
                    r_pos = compute_radius(pos)
                    pos[i, dim] = max(-0.499999, old - step)
                    r_neg = compute_radius(pos)
                    pos[i, dim] = old
                    if r_pos > r_cur + 1e-13:
                        pos[i, dim] = min(0.499999, old + step)
                        improved = True
                    elif r_neg > r_cur + 1e-13:
                        pos[i, dim] = max(-0.499999, old - step)
                        improved = True
            if not improved:
                step *= 0.8
                if step < 1e-7:
                    break
        return pos

    def full_pipeline(positions):
        """Multi-stage optimization pipeline."""
        if remaining() < 10:
            return positions

        # Stage 1: Smooth surrogate with beta annealing
        if remaining() > 30:
            pos = local_optimize_smooth(positions, beta_start=5.0, beta_end=80.0, n_beta=5, maxiter_per=80)
        else:
            pos = positions

        # Stage 2: Nesterov active-set gradient ascent
        if remaining() > 15:
            pos = nesterov_active_set_gradient_ascent(pos, maxiter=300, lr_start=0.0015, lr_end=0.00005)

        # Stage 3: Direct L-BFGS-B polish
        if remaining() > 8:
            pos = local_optimize_direct(pos, maxiter=150)

        # Stage 4: Coordinate descent
        if remaining() > 3:
            pos = coordinate_descent(pos, maxiter=40)

        return pos

    best_positions = None
    best_radius = -1.0

    def update_best(positions):
        nonlocal best_positions, best_radius
        r = compute_radius(positions)
        if r > best_radius:
            best_radius = r
            best_positions = positions.copy()
        return r

    # Phase 1: Multiple initializations
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_like),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
    ]

    for name, init_func in initializations:
        if remaining() < 15:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            pos_opt = full_pipeline(positions)
            update_best(pos_opt)
        except Exception:
            continue

    # Phase 2: Differential evolution for global search
    if remaining() > 35:
        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_de(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.differential_evolution(obj_de, bounds, method='de',
                                             popsize=10, maxiter=20, seed=0,
                                             tol=1e-10, mutation=(0.4, 1.0), recombination=0.7,
                                             polish=False, updating='immediate')
            pos_de = np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
            if remaining() > 15:
                pos_polished = full_pipeline(pos_de)
                update_best(pos_polished)
            else:
                update_best(pos_de)
        except Exception:
            pass

    # Phase 3: Perturbation refinement
    if best_positions is not None and remaining() > 15:
        for perturb_seed in range(3):
            if remaining() < 10:
                break
            rng = np.random.RandomState(1000 + perturb_seed)
            perturbed = best_positions + rng.normal(0, 0.01, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            pos_opt = full_pipeline(perturbed)
            update_best(pos_opt)

    # Final polish
    if best_positions is not None and remaining() > 8:
        best_positions = local_optimize_direct(best_positions, maxiter=100)
        if remaining() > 3:
            best_positions = coordinate_descent(best_positions, maxiter=30)
        update_best(best_positions)

    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    return best_positions
