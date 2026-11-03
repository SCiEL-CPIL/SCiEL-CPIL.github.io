"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_vectorized(centers):
        """
        Vectorized computation of maximum non-overlapping radii from centers.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)

        # Boundary constraints (vectorized)
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        # Pairwise constraints (vectorized)
        for i in range(n):
            dx = centers[i, 0] - centers[i+1:, 0]
            dy = centers[i, 1] - centers[i+1:, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            max_rs = dists / 2.0
            radii[i] = min(radii[i], np.min(max_rs) if len(max_rs) > 0 else radii[i])
            radii[i+1:] = np.minimum(radii[i+1:], max_rs)

        return radii

    def eval_packing(centers):
        """Evaluate the sum of radii for given centers."""
        radii = compute_radii_vectorized(centers)
        return np.sum(radii), radii

    def simulated_annealing(centers, max_iters=50000, T_start=0.05, T_end=0.001, seed=0):
        """
        Simulated annealing to optimize circle center positions.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)

        best = current.copy()
        best_score = current_score

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break

            # Cooling schedule: geometric
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))

            # Pick a random circle to move
            i = rng.randint(N)
            # Adaptive step size
            step = 0.02 * (T / T_start) ** 0.5 + 0.002
            dx = rng.normal(0, step)
            dy = rng.normal(0, step)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)
            delta = new_score - current_score

            # Acceptance criterion
            if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def local_search(centers, max_iters=3000, step_size=0.01, seed=0):
        """
        Best-fit local search: only accept improvements.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)

            if new_score > current_score + 1e-12:
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def grid_init():
        """Structured grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def hex_init():
        """Hexagonal-like initialization."""
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) < N:
                    x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                    y = 0.1 + j * 0.17
                    if x < 0.9 and y < 0.9:
                        centers.append([x, y])
        if len(centers) < N:
            # Fill remaining with random
            rng = np.random.RandomState(42)
            while len(centers) < N:
                centers.append(list(rng.uniform(0.1, 0.9, 2)))
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    # Main optimization loop
    best_result = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    inits.append(('grid', grid_init()))
    inits.append(('hex', hex_init()))
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131, 151, 173]:
        inits.append((f'rand_{s}', random_init(s)))

    # Run SA + local search from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 10:
            break

        # Phase 1: Simulated Annealing
        sa_centers, sa_score = simulated_annealing(
            init_centers,
            max_iters=20000,
            T_start=0.04,
            T_end=0.002,
            seed=hash(init_name) % 10000
        )

        # Phase 2: Local search polish
        ls_centers, ls_score = local_search(
            sa_centers,
            max_iters=2000,
            step_size=0.008,
            seed=hash(init_name) % 10000 + 1
        )

        if ls_score > best_score:
            best_score = ls_score
            radii = compute_radii_vectorized(ls_centers)
            best_result = np.column_stack([ls_centers, radii])

        # Also check SA result directly (in case local search degraded it)
        if sa_score > best_score:
            best_score = sa_score
            radii = compute_radii_vectorized(sa_centers)
            best_result = np.column_stack([sa_centers, radii])

    # If no valid result, fall back to grid
    if best_result is None:
        centers = grid_init()
        radii = compute_radii_vectorized(centers)
        best_result = np.column_stack([centers, radii])

    # Final refinement: multiple SA restarts from best solution
    if best_result is not None and time.time() < deadline - 15:
        for trial in range(8):
            if time.time() > deadline - 5:
                break

            # Perturb best solution
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.02, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA from perturbed
            sa_centers, sa_score = simulated_annealing(
                perturbed,
                max_iters=10000,
                T_start=0.02,
                T_end=0.001,
                seed=1000 + trial
            )

            # Local search
            ls_centers, ls_score = local_search(
                sa_centers,
                max_iters=1000,
                step_size=0.005,
                seed=2000 + trial
            )

            if ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Ensure final validity with small margin
    if best_result is not None:
        # Apply tiny margin to ensure strict validity
        radii = compute_radii_vectorized(best_result[:, :2])
        radii = np.maximum(0, radii - 1e-9)
        best_result = np.column_stack([best_result[:, :2], radii])

        if not is_valid(best_result, tol=1e-6):
            # Fallback: conservative grid
            centers = grid_init()
            radii = compute_radii_vectorized(centers, margin=0.001) if hasattr(compute_radii_vectorized, 'margin') else compute_radii_vectorized(centers)
            radii = np.maximum(0, radii - 0.001)
            best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
