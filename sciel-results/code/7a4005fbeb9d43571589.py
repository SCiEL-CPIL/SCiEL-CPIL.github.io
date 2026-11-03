"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_from_centers(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii using
        all pairwise distances and boundary distances.
        Returns array of radii.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)

        # Compute boundary distances
        for i in range(n):
            d_left = centers[i, 0]
            d_right = 1.0 - centers[i, 0]
            d_bottom = centers[i, 1]
            d_top = 1.0 - centers[i, 1]
            radii[i] = min(d_left, d_right, d_bottom, d_top)

        # Enforce non-overlap for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = dist / 2.0
                radii[i] = min(radii[i], max_r)
                radii[j] = min(radii[j], max_r)

        if margin > 0:
            radii = np.maximum(0.0, radii - margin)

        return radii

    def max_radius_for_circle(x, y, others_xy, others_r):
        """
        Given a candidate center (x, y) and positions/radii of other circles,
        return the maximum radius such that the circle stays inside [0,1]^2
        and does not overlap any other circle.
        """
        wall_clear = min(x, 1.0 - x, y, 1.0 - y)
        if wall_clear <= 0:
            return 0.0
        if len(others_xy) > 0:
            dx = x - others_xy[:, 0]
            dy = y - others_xy[:, 1]
            dists = np.sqrt(dx * dx + dy * dy)
            circle_clear = np.min(dists - others_r)
            return max(0.0, min(wall_clear, circle_clear))
        return max(0.0, wall_clear)

    def eval_packing(pos):
        """
        Evaluate a packing configuration (N x 3 array of [x, y, r]).
        Returns (total_radius, is_valid).
        """
        xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
        total_r = np.sum(rs)
        if np.any(rs < 0) or np.any(xs < 0) or np.any(xs > 1) or np.any(ys < 0) or np.any(ys > 1):
            return total_r, False
        if np.any(rs > xs) or np.any(rs > 1 - xs) or np.any(rs > ys) or np.any(rs > 1 - ys):
            return total_r, False
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                if d2 < (rs[i] + rs[j]) ** 2 - 1e-9:
                    return total_r, False
        return total_r, True

    def best_fit_local_search(pos, rng, deadline, max_rounds=30):
        """
        Iteratively for each circle, find the best position and maximum radius
        that fits given all other circles.
        """
        pos = pos.copy()
        current_total, _ = eval_packing(pos)

        for round_num in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False
            order = rng.permutation(N)

            for i in order:
                if time.time() > deadline:
                    break

                others_idx = [j for j in range(N) if j != i]
                others_xy = pos[others_idx, :2]
                others_r = pos[others_idx, 2]

                best_r = pos[i, 2]
                best_x, best_y = pos[i, 0], pos[i, 1]
                best_candidate_r = best_r
                best_candidate_pos = (best_x, best_y)

                cx, cy = pos[i, 0], pos[i, 1]
                search_radius = max(0.15, 2 * pos[i, 2] + 0.05)

                # Multi-scale grid search
                for scale in [1.0, 0.5, 0.25]:
                    n_pts = int(20 * scale)
                    if n_pts < 6:
                        n_pts = 6

                    grid_x = np.linspace(best_x - search_radius * scale,
                                         best_x + search_radius * scale, n_pts)
                    grid_y = np.linspace(best_y - search_radius * scale,
                                         best_y + search_radius * scale, n_pts)

                    gx, gy = np.meshgrid(grid_x, grid_y)
                    gx = gx.ravel()
                    gy = gy.ravel()

                    valid_mask = (gx >= 0) & (gx <= 1) & (gy >= 0) & (gy <= 1)
                    gx = gx[valid_mask]
                    gy = gy[valid_mask]

                    if len(gx) == 0:
                        continue

                    # Vectorize radius computation
                    for k in range(len(gx)):
                        r = max_radius_for_circle(gx[k], gy[k], others_xy, others_r)
                        if r > best_candidate_r + 1e-10:
                            best_candidate_r = r
                            best_candidate_pos = (gx[k], gy[k])

                # Also try random positions
                n_random = 30
                rand_x = rng.uniform(0, 1, n_random)
                rand_y = rng.uniform(0, 1, n_random)
                for k in range(n_random):
                    r = max_radius_for_circle(rand_x[k], rand_y[k], others_xy, others_r)
                    if r > best_candidate_r + 1e-10:
                        best_candidate_r = r
                        best_candidate_pos = (rand_x[k], rand_y[k])

                if best_candidate_r > best_r + 1e-10:
                    pos[i, 0] = best_candidate_pos[0]
                    pos[i, 1] = best_candidate_pos[1]
                    pos[i, 2] = best_candidate_r
                    improved = True

            if not improved:
                break

        # Final validity check and clip
        for i in range(N):
            pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)

        return pos

    def soft_min(values, beta=100.0):
        """
        Smooth approximation of min using log-sum-exp (negative).
        """
        values = np.asarray(values, dtype=np.float64)
        m = np.min(values)
        if m < -1000:
            return m
        shifted = np.clip(beta * (values - m), -700, 700)
        return m - np.log(np.sum(np.exp(shifted))) / beta

    def soft_min_pairwise(centers, beta=100.0):
        """
        Compute smooth approximation of radii using soft-min.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)

        for i in range(n):
            constraints = np.array([
                centers[i, 0],
                1.0 - centers[i, 0],
                centers[i, 1],
                1.0 - centers[i, 1]
            ])

            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                constraints = np.append(constraints, dist / 2.0)

            radii[i] = soft_min(constraints, beta)

        return radii

    def objective_smooth(centers_flat, beta=100.0):
        """
        Smooth objective using soft-min for the radii computation.
        """
        centers = centers_flat.reshape(-1, 2)
        radii = soft_min_pairwise(centers, beta)
        return -np.sum(radii)

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    def local_search(centers, max_iters=5000, step_size=0.02, seed=0):
        """
        Discrete best-fit local search: try small perturbations to centers,
        accept if it improves the sum of radii.
        """
        rng = np.random.RandomState(seed)
        current_centers = centers.copy()
        current_radii = compute_radii_from_centers(current_centers)
        current_score = np.sum(current_radii)

        best_centers = current_centers.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)

            new_centers = current_centers.copy()
            new_centers[i, 0] = np.clip(new_centers[i, 0] + dx, 0.02, 0.98)
            new_centers[i, 1] = np.clip(new_centers[i, 1] + dy, 0.02, 0.98)

            new_radii = compute_radii_from_centers(new_centers)
            new_score = np.sum(new_radii)

            if new_score > current_score + 1e-10:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()

        return best_centers

    def simulated_annealing(centers, max_iters=10000, t_start=0.05, t_end=0.001, seed=0):
        """
        Simulated annealing to explore the configuration space stochastically.
        """
        rng = np.random.RandomState(seed)
        current_centers = centers.copy()
        current_radii = compute_radii_from_centers(current_centers)
        current_score = np.sum(current_radii)

        best_centers = current_centers.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            t = t_start * (t_end / t_start) ** (it / max_iters)

            i = rng.randint(N)
            dx = rng.uniform(-0.05, 0.05)
            dy = rng.uniform(-0.05, 0.05)

            new_centers = current_centers.copy()
            new_centers[i, 0] = np.clip(new_centers[i, 0] + dx, 0.02, 0.98)
            new_centers[i, 1] = np.clip(new_centers[i, 1] + dy, 0.02, 0.98)

            new_radii = compute_radii_from_centers(new_centers)
            new_score = np.sum(new_radii)

            delta = new_score - current_score

            if delta > 0 or rng.random() < np.exp(delta / t):
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()

        return best_centers

    def gradient_polish(centers, max_iters=200, use_smooth=True):
        """
        Final high-precision polish using L-BFGS-B with tighter tolerances.
        """
        try:
            if use_smooth:
                for beta in [100.0, 200.0, 500.0]:
                    result = minimize(
                        lambda x: objective_smooth(x, beta),
                        centers.flatten(),
                        method='L-BFGS-B',
                        bounds=[(0.01, 0.99)] * (2 * N),
                        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    opt_centers = result.x.reshape(-1, 2)
                    radii = compute_radii_from_centers(opt_centers)
                    x_full = np.column_stack([opt_centers, radii])
                    if is_valid(x_full, tol=1e-6):
                        return x_full
            else:
                def objective_only(centers_flat):
                    centers = centers_flat.reshape(-1, 2)
                    radii = compute_radii_from_centers(centers)
                    return -np.sum(radii)

                result = minimize(
                    objective_only,
                    centers.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0.01, 0.99)] * (2 * N),
                    options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_centers = result.x.reshape(-1, 2)
                radii = compute_radii_from_centers(opt_centers)
                x_full = np.column_stack([opt_centers, radii])
                if is_valid(x_full, tol=1e-6):
                    return x_full
        except Exception:
            pass
        return None

    # Strategy: Multiple initializations with best-fit local search + gradient polish
    best_result = None
    best_score = -np.inf

    inits = []
    inits.append(initial_grid_centers())

    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131]:
        inits.append(random_centers(seed))

    # Hexagonal-like pattern
    hex_centers = []
    for i in range(6):
        for j in range(5):
            if len(hex_centers) < N:
                x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                y = 0.1 + j * 0.17
                if x < 0.9 and y < 0.9:
                    hex_centers.append([x, y])
    if len(hex_centers) >= N:
        inits.append(np.array(hex_centers[:N]))

    # Optimize each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 20:
            break

        # Step 1: Best-fit local search (from inspiration)
        pos = np.column_stack([init_centers, compute_radii_from_centers(init_centers)])
        rng = np.random.RandomState(init_idx)
        bf_pos = best_fit_local_search(pos, rng, deadline - 10, max_rounds=20)
        bf_score = np.sum(bf_pos[:, 2])

        # Step 2: Also try the discrete local search
        ls_centers = local_search(init_centers, max_iters=2000, step_size=0.03, seed=init_idx)
        ls_radii = compute_radii_from_centers(ls_centers)
        ls_score = np.sum(ls_radii)

        # Step 3: Simulated annealing
        sa_centers = simulated_annealing(ls_centers, max_iters=3000, t_start=0.03, t_end=0.001, seed=init_idx)
        sa_radii = compute_radii_from_centers(sa_centers)
        sa_score = np.sum(sa_radii)

        # Pick the best so far
        candidates = []
        if bf_score > 0:
            candidates.append((bf_score, bf_pos[:, :2]))
        if ls_score > 0:
            candidates.append((ls_score, ls_centers))
        if sa_score > 0:
            candidates.append((sa_score, sa_centers))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            current_centers = candidates[0][1]
        else:
            current_centers = ls_centers

        # Step 4: Gradient polish with smooth objective
        polished = gradient_polish(current_centers, max_iters=150, use_smooth=True)

        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
        else:
            polished = gradient_polish(current_centers, max_iters=100, use_smooth=False)
            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            else:
                x_ls = np.column_stack([ls_centers, ls_radii])
                if is_valid(x_ls, tol=1e-6) and ls_score > best_score:
                    best_score = ls_score
                    best_result = x_ls.copy()

    # If no valid result found, fall back to grid
    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])

    # Final refinement: more search from best result
    if best_result is not None and time.time() < deadline - 25:
        x_ref = best_result.copy()
        for trial in range(6):
            if time.time() > deadline - 10:
                break

            rng = np.random.RandomState(100 + trial)
            perturbed = x_ref[:, :2].copy()
            perturbed += rng.normal(0, 0.015, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.02, 0.98)

            # Best-fit local search
            pos = np.column_stack([perturbed, compute_radii_from_centers(perturbed)])
            bf_pos = best_fit_local_search(pos, rng, deadline - 5, max_rounds=15)
            bf_score = np.sum(bf_pos[:, 2])

            # Discrete local search
            ls_centers = local_search(perturbed, max_iters=1500, step_size=0.02, seed=200 + trial)
            ls_radii = compute_radii_from_centers(ls_centers)
            ls_score = np.sum(ls_radii)

            # SA
            sa_centers = simulated_annealing(ls_centers, max_iters=2000, t_start=0.02, t_end=0.001, seed=300 + trial)
            sa_radii = compute_radii_from_centers(sa_centers)
            sa_score = np.sum(sa_radii)

            candidates = []
            if bf_score > 0:
                candidates.append((bf_score, bf_pos[:, :2]))
            if ls_score > 0:
                candidates.append((ls_score, ls_centers))
            if sa_score > 0:
                candidates.append((sa_score, sa_centers))

            if candidates:
                candidates.sort(key=lambda x: -x[0])
                current_centers = candidates[0][1]
            else:
                current_centers = ls_centers

            polished = gradient_polish(current_centers, max_iters=100, use_smooth=True)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            else:
                polished = gradient_polish(current_centers, max_iters=80, use_smooth=False)
                if polished is not None:
                    p_score = np.sum(polished[:, 2])
                    if p_score > best_score:
                        best_score = p_score
                        best_result = polished.copy()
                elif is_valid(np.column_stack([ls_centers, ls_radii]), tol=1e-6) and ls_score > best_score:
                    best_score = ls_score
                    best_result = np.column_stack([ls_centers, ls_radii]).copy()

    # Ensure final validity
    if best_result is None or not is_valid(best_result, tol=1e-6):
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
