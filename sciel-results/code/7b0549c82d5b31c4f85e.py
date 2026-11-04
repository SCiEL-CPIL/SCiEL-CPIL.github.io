"""Combined strategy: Vectorized smooth-surrogate (log-sum-exp) with analytic gradients,
beta-annealing, active-set gradient ascent on the exact min-distance,
plus targeted perturbation and final high-precision polish. For 50 equal spheres in [-1/2,1/2]^3."""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


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
        # Wall distances
        min_wall = np.min(0.5 - np.abs(positions))
        # Pairwise distances using vectorized approach
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
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

    def active_set_gradient_ascent(c_init, maxiter=500, lr_start=0.003, lr_end=0.00005):
        """
        Nesterov-accelerated gradient ascent on the exact min-distance objective.
        Uses active-set subgradient with momentum for efficiency.
        """
        if remaining() < 3:
            return c_init

        c = c_init.copy()
        v = np.zeros_like(c)
        best_c = c.copy()
        best_r = compute_radius(c)
        momentum = 0.7

        for it in range(maxiter):
            if remaining() < 2:
                break

            lr = lr_start * (lr_end / lr_start) ** (it / max(1, maxiter))

            # Compute distances
            diff = c[:, None, :] - c[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)

            # Wall distances per point per dimension
            wall_d = 0.5 - np.abs(c)  # (N,3)

            # Current minimum radius
            min_pair = np.min(d) / 2.0
            min_wall = np.min(wall_d)
            current_min = min(min_pair, min_wall)

            # Active set threshold
            threshold = max(1.3 * current_min, 0.005)

            # Pairwise active set (upper triangle only)
            active_pair_mask = (d < 2.0 * threshold)
            np.fill_diagonal(active_pair_mask, False)
            # Only use upper triangle to avoid double counting
            tri_mask = np.triu(np.ones((N, N), dtype=bool), 1)
            active_pair_mask = active_pair_mask & tri_mask
            ai, aj = np.where(active_pair_mask)

            grad = np.zeros_like(c)
            if len(ai) > 0:
                d_active = d[ai, aj]
                d_safe = np.maximum(d_active, 1e-12)
                # Weight by proximity: closer pairs get more weight
                weights = np.maximum(1.0 - d_active / (2.0 * threshold), 0.1)
                unit_diffs = (c[ai] - c[aj]) / d_safe[:, None]
                weighted = weights[:, None] * unit_diffs
                np.add.at(grad, ai, weighted)
                np.add.at(grad, aj, -weighted)

            # Wall active set (vectorized)
            wall_active = wall_d < threshold
            # For each active wall, add gradient pointing inward
            # wall_d[i,dim] = 0.5 - |c[i,dim]|
            # If c[i,dim] > 0, inward is -x direction (decrease c)
            # If c[i,dim] < 0, inward is +x direction (increase c)
            wall_weights = np.where(wall_active, np.maximum(1.0 - wall_d / threshold, 0.1), 0.0)
            sign = np.sign(c)
            sign[sign == 0] = 1.0
            # Gradient for walls: push away from wall
            # If c > 0, wall at +0.5, push in -x direction: grad -= 1
            # If c < 0, wall at -0.5, push in +x direction: grad += 1
            grad -= wall_weights * sign

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-15:
                grad = grad / grad_norm

            # Nesterov/momentum update
            v = momentum * v + lr * grad
            c_new = c + v
            c_new = np.clip(c_new, -0.499999, 0.499999)

            # Check improvement
            r_new = compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new
            else:
                # Reduce momentum on non-improvement
                momentum = max(0.3, momentum * 0.95)
                c = c_new

        return best_c

    def local_optimize_smooth(c_init, beta_start=3.0, beta_end=150.0, n_beta=8, maxiter_per=120):
        """L-BFGS-B on smooth surrogate with beta annealing."""
        if remaining() < 5:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)
        # Use geometric progression for beta for better annealing
        betas = np.geomspace(beta_start, beta_end, n_beta)
        pos = c_init.flatten()

        for beta in betas:
            if remaining() < 4:
                break

            def obj_flat(x):
                val, _ = smooth_objective_and_grad(x, beta)
                return val

            def grad_flat(x):
                _, grad = smooth_objective_and_grad(x, beta)
                return grad

            try:
                res = opt.minimize(obj_flat, pos, jac=grad_flat, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': maxiter_per, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
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
                               bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10, 'maxls': 50})
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

    def fine_polish(positions, maxiter=100):
        """High-precision active-set polish with adaptive step size."""
        if remaining() < 3:
            return positions
        pos = positions.copy()
        step = 0.002
        best_r = compute_radius(pos)
        for it in range(maxiter):
            if remaining() < 2:
                break
            improved = False
            # Compute active set
            diff = pos[:, None, :] - pos[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            wall_d = 0.5 - np.abs(pos)
            min_pair = np.min(d) / 2.0
            min_wall = np.min(wall_d)
            current_min = min(min_pair, min_wall)
            threshold = max(1.2 * current_min, 0.002)

            # Find active constraints
            active_pairs = (d < 2.0 * threshold)
            np.fill_diagonal(active_pairs, False)
            active_pairs = active_pairs & np.triu(np.ones((N, N), dtype=bool), 1)
            ai, aj = np.where(active_pairs)

            active_walls = wall_d < threshold

            # Try small moves along gradient directions
            grad = np.zeros_like(pos)
            if len(ai) > 0:
                d_active = d[ai, aj]
                d_safe = np.maximum(d_active, 1e-12)
                unit_diffs = (pos[ai] - pos[aj]) / d_safe[:, None]
                np.add.at(grad, ai, unit_diffs)
                np.add.at(grad, aj, -unit_diffs)

            wall_weights = np.where(active_walls, 1.0, 0.0)
            sign = np.sign(pos)
            sign[sign == 0] = 1.0
            grad -= wall_weights * sign

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            grad = grad / grad_norm

            # Try the step
            pos_trial = pos + step * grad
            pos_trial = np.clip(pos_trial, -0.499999, 0.499999)
            r_trial = compute_radius(pos_trial)
            if r_trial > best_r + 1e-15:
                pos = pos_trial
                best_r = r_trial
                improved = True
            else:
                # Try half step
                pos_trial = pos + 0.5 * step * grad
                pos_trial = np.clip(pos_trial, -0.499999, 0.499999)
                r_trial = compute_radius(pos_trial)
                if r_trial > best_r + 1e-15:
                    pos = pos_trial
                    best_r = r_trial
                    improved = True
                else:
                    step *= 0.7
                    if step < 1e-9:
                        break
            if not improved:
                step *= 0.85
                if step < 1e-9:
                    break
        return pos

    def full_pipeline(positions):
        """Multi-stage optimization pipeline."""
        if remaining() < 10:
            return positions

        # Stage 1: Smooth surrogate with beta annealing
        if remaining() > 25:
            pos = local_optimize_smooth(positions, beta_start=3.0, beta_end=150.0, n_beta=8, maxiter_per=100)
        else:
            pos = positions

        # Stage 2: Active-set gradient ascent
        if remaining() > 15:
            pos = active_set_gradient_ascent(pos, maxiter=400, lr_start=0.003, lr_end=0.00005)

        # Stage 3: Direct L-BFGS-B polish
        if remaining() > 10:
            pos = local_optimize_direct(pos, maxiter=200)

        # Stage 4: Fine polish
        if remaining() > 5:
            pos = fine_polish(pos, maxiter=80)

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

    # Phase 2: Targeted perturbation from best solution
    if best_positions is not None and remaining() > 30:
        for pert_seed in range(5):
            if remaining() < 15:
                break
            rng = np.random.RandomState(2000 + pert_seed)
            # Scale perturbation by current radius
            scale = best_radius * 0.3
            perturbed = best_positions + rng.normal(0, scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            pos_opt = full_pipeline(perturbed)
            update_best(pos_opt)

    # Phase 3: Smaller perturbations for fine-tuning
    if best_positions is not None and remaining() > 20:
        for perturb_seed in range(4):
            if remaining() < 12:
                break
            rng = np.random.RandomState(3000 + perturb_seed)
            scale = best_radius * 0.1
            perturbed = best_positions + rng.normal(0, scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            pos_opt = full_pipeline(perturbed)
            update_best(pos_opt)

    # Final high-precision polish
    if best_positions is not None and remaining() > 10:
        best_positions = local_optimize_direct(best_positions, maxiter=150)
        if remaining() > 5:
            best_positions = fine_polish(best_positions, maxiter=100)
        update_best(best_positions)

    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    return best_positions
