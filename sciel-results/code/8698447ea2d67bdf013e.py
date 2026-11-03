"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


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
        diff = c[:, np.newaxis, :] - c[np.newaxis, :]
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
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :]
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
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def exact_subgradient(c):
        """
        Compute an exact subgradient for the nonsmooth min objective.
        Returns gradient pointing in direction that increases the minimum constraint.
        """
        grad = np.zeros_like(c)
        
        # Compute all constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall = np.min(dist_to_walls)
        
        current_r = min(min_pair / 2.0, min_wall)
        eps = 1e-8
        
        # Pairwise constraints
        active_pairs = d < (min_pair + eps)
        rows, cols = np.where(active_pairs)
        
        for r_idx, c_idx in zip(rows, cols):
            if r_idx == c_idx:
                continue
            vec = c[r_idx] - c[c_idx]
            dist = np.linalg.norm(vec)
            if dist > 1e-12:
                grad[r_idx] += vec / dist
                grad[c_idx] -= vec / dist
        
        # Wall constraints
        for i in range(N):
            if dist_to_walls[i] < current_r + eps:
                x, y, z = c[i]
                dx = 0.5 - abs(x)
                dy = 0.5 - abs(y)
                dz = 0.5 - abs(z)
                
                if dx <= dy and dx <= dz:
                    grad[i, 0] += 1.0 if x >= 0 else -1.0
                elif dy <= dx and dy <= dz:
                    grad[i, 1] += 1.0 if y >= 0 else -1.0
                else:
                    grad[i, 2] += 1.0 if z >= 0 else -1.0
        
        return grad

    def final_polish(c_init, max_iters=200):
        """
        Final high-precision polishing using exact subgradient with line search.
        """
        c = c_init.copy()
        best_r = get_radius(c)
        best_c = c.copy()
        
        step_size = 0.001
        stagnation = 0
        
        for i in range(max_iters):
            if remaining() < 1.0:
                break
            
            grad = exact_subgradient(c)
            norm = np.linalg.norm(grad)
            if norm < 1e-14:
                break
            grad /= norm
            
            # Backtracking line search
            for attempt in range(20):
                new_c = c + step_size * grad
                new_c = np.clip(new_c, -0.499, 0.499)
                new_r = get_radius(new_c)
                
                if new_r > best_r + 1e-12:
                    c = new_c
                    best_r = new_r
                    best_c = c.copy()
                    stagnation = 0
                    break
                else:
                    step_size *= 0.7
            
            else:
                # No improvement, try smaller step
                step_size *= 0.5
                if step_size < 1e-8:
                    break
                stagnation += 1
                if stagnation > 50:
                    break
        
        return best_c, best_r

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

        # Final polish with exact subgradient
        if remaining() > 3.0:
            c_polished, r_polished = final_polish(c_best)
            if r_polished > r_best:
                c_best = c_polished.copy()
                r_best = r_polished

        return c_best, r_best

    # Generate diverse initializations
    np.random.seed(42)
    init_configs = []

    # 1. Jittered grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    init_configs.append(grid + np.random.normal(0, 0.02, size=grid.shape))

    # 2. FCC-inspired initialization
    # FCC lattice in a cube: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_points.append([i/3 - 0.5, j/3 - 0.5, k/3 - 0.5])
                fcc_points.append([i/3 - 0.5, j/3 - 0.5 + 0.5/3, k/3 - 0.5 + 0.5/3])
                fcc_points.append([i/3 - 0.5 + 0.5/3, j/3 - 0.5, k/3 - 0.5 + 0.5/3])
                fcc_points.append([i/3 - 0.5 + 0.5/3, j/3 - 0.5 + 0.5/3, k/3 - 0.5])
    fcc_points = np.array(fcc_points[:N])
    init_configs.append(fcc_points + np.random.normal(0, 0.01, size=fcc_points.shape))

    # 3. Random initializations
    for _ in range(5):
        rand_c = np.random.uniform(-CUBE_HALF * 0.8, CUBE_HALF * 0.8, size=(N, 3))
        init_configs.append(rand_c)

    # Run optimization on each initialization
    best_centers = None
    best_r = 0.0

    for idx, config in enumerate(init_configs):
        if remaining() < 5.0:
            break

        c, r = annealed_optimization(config)

        if r > best_r:
            best_r = r
            best_centers = c.copy()

    if best_centers is None:
        return init_configs[0]

    return best_centers
# EVOLVE-BLOCK-END
