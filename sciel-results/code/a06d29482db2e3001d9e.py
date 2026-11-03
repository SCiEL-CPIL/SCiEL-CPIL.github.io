"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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

    def delaunay_radii(centers):
        """
        Compute radii using Delaunay triangulation neighbor distances.
        For each circle, the radius is limited by half the distance to its
        nearest neighbor (from Delaunay) and by boundary distance.
        This is a tighter constraint than using ALL pairs.
        """
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)

        # Boundary constraints
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        # Try Delaunay-based nearest neighbor constraint
        try:
            if n >= 3:
                tri = Delaunay(centers)
                # For each point, find distances to its Delaunay neighbors
                for i in range(n):
                    # Get simplices containing point i
                    simplices_with_i = tri.find_simplex(centers[i:i+1, :])
                    if simplices_with_i[0] >= 0:
                        # Collect unique neighbors from these simplices
                        neighbors = set()
                        for s in simplices_with_i:
                            for v in tri.simplices[s]:
                                if v != i:
                                    neighbors.add(v)
                        # Find minimum distance to neighbors
                        if neighbors:
                            min_dist = np.inf
                            for j in neighbors:
                                dx = centers[i, 0] - centers[j, 0]
                                dy = centers[i, 1] - centers[j, 1]
                                d = np.sqrt(dx*dx + dy*dy)
                                min_dist = min(min_dist, d)
                            radii[i] = min(radii[i], min_dist / 2.0)
        except Exception:
            pass

        # Fallback: ensure pairwise non-overlap with full check
        # The Delaunay approach may miss some constraints, so we need to verify
        # and fix any violations
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                if radii[i] + radii[j] > dist - 1e-10:
                    # Need to reduce radii
                    # Proportional reduction
                    total = radii[i] + radii[j]
                    if total > 0:
                        scale = (dist - 1e-10) / total
                        radii[i] *= scale
                        radii[j] *= scale

        return radii

    def objective_smooth(centers_flat, beta=100.0):
        """
        Smooth objective using soft-min (log-sum-exp) to approximate the
        max-based radius constraint. This provides smooth gradients for L-BFGS-B.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        # Compute all pairwise distances and boundary distances
        # For each circle i, the radius is min of:
        #   - boundary distances
        #   - d_ij / 2 for all j != i
        # We use soft-min to make this differentiable

        # Boundary constraints per circle
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]

        # For soft-min, we compute -soft-min = soft-max of negatives
        # soft-min(x) = -(1/beta) * log(sum(exp(-beta*x_k)))
        # So radius_i = soft-min(boundary_dists_i, pairwise_dists_i/2)

        total_sum = 0.0
        for i in range(n):
            # Collect all constraints for circle i
            constraints = [d_left[i], d_right[i], d_bottom[i], d_top[i]]
            for j in range(n):
                if j != i:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy) / 2.0
                    constraints.append(d)

            # Soft-min of constraints
            c = np.array(constraints)
            # Use log-sum-exp trick for numerical stability
            c_min = np.min(c)
            log_sum = np.log(np.sum(np.exp(-beta * (c - c_min))))
            soft_min_val = c_min - np.log(log_sum) / beta

            total_sum += soft_min_val

        return -total_sum  # Negate for minimization

    def gradient_polish(centers, max_iters=200, beta=100.0):
        """
        Final high-precision polish using L-BFGS-B with smooth objective.
        """
        try:
            result = minimize(
                lambda x: objective_smooth(x, beta=beta),
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_vectorized(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None

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

    # Run SA + local search + gradient polish from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 10:
            break

        # Phase 1: Simulated Annealing
        sa_centers, sa_score = simulated_annealing(
            init_centers,
            max_iters=15000,
            T_start=0.04,
            T_end=0.002,
            seed=hash(init_name) % 10000
        )

        # Phase 2: Local search polish
        ls_centers, ls_score = local_search(
            sa_centers,
            max_iters=1500,
            step_size=0.008,
            seed=hash(init_name) % 10000 + 1
        )

        # Phase 3: Gradient polish with smooth objective
        polished = gradient_polish(ls_centers, max_iters=150, beta=100.0)

        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
        else:
            # Use local search result if polish failed
            if ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

        # Also check SA result directly
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
    if best_result is not None and time.time() < deadline - 20:
        for trial in range(10):
            if time.time() > deadline - 5:
                break

            # Perturb best solution
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.015, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA from perturbed
            sa_centers, sa_score = simulated_annealing(
                perturbed,
                max_iters=8000,
                T_start=0.02,
                T_end=0.001,
                seed=1000 + trial
            )

            # Local search
            ls_centers, ls_score = local_search(
                sa_centers,
                max_iters=800,
                step_size=0.005,
                seed=2000 + trial
            )

            # Gradient polish
            polished = gradient_polish(ls_centers, max_iters=100, beta=100.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Second pass: more local search from best result with finer steps
    if best_result is not None and time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break

            # Slight perturbation
            rng = np.random.RandomState(3000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.008, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # Fine local search
            ls_centers, ls_score = local_search(
                perturbed,
                max_iters=1000,
                step_size=0.004,
                seed=4000 + trial
            )

            # Gradient polish
            polished = gradient_polish(ls_centers, max_iters=100, beta=150.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
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
            radii = compute_radii_vectorized(centers)
            radii = np.maximum(0, radii - 0.001)
            best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
