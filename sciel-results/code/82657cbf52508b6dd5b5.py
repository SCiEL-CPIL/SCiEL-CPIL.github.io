"""Direct multi-objective optimizer for WFG7 using Tchebycheff decomposition with adaptive local search."""
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for WFG7 using Tchebycheff decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def _is_non_dominated(F, f):
            """Check if f is not dominated by any point in F."""
            for ef in F:
                if _dominates(ef, f):
                    return False
            return True

        def _remove_dominated(F, f):
            """Remove points from F that are dominated by f."""
            return [ef for ef in F if not _dominates(f, ef)]

        def _hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points below reference point."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            n = len(valid)

            total_hv = 0.0
            for i in range(n):
                x1, y1, z1 = valid[i]
                x2 = valid[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1

                # Compute 2D HV in y-z plane for points i..n-1
                yz_pts = [(valid[j][1], valid[j][2]) for j in range(i, n)
                          if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
                if not yz_pts:
                    continue

                yz_pts.sort(key=lambda p: p[0])
                m = len(yz_pts)

                # Sweep from high y to low y
                min_z = ref[2]
                for j in range(m - 1, -1, -1):
                    y, z = yz_pts[j]
                    if z < min_z:
                        min_z = z
                    y_next = yz_pts[j + 1][0] if j + 1 < m else ref[1]
                    dy = y_next - y
                    dz = ref[2] - min_z
                    if dy > 0 and dz > 0:
                        total_hv += dx * dy * dz

            return total_hv

        # Generate Tchebycheff weight vectors for 3 objectives
        def _generate_weights(n_obj, n_weights):
            """Generate evenly distributed weight vectors on the simplex."""
            weights = []
            if n_obj == 1:
                weights.append(np.array([1.0]))
            elif n_obj == 2:
                for i in range(n_weights):
                    w1 = i / (n_weights - 1) if n_weights > 1 else 0.5
                    weights.append(np.array([w1, 1 - w1]))
            elif n_obj == 3:
                # Grid on 2D simplex
                n_per_edge = int(np.ceil(n_weights ** 0.5))
                for i in range(n_per_edge + 1):
                    for j in range(n_per_edge + 1 - i):
                        k = n_per_edge - i - j
                        w = np.array([i, j, k], dtype=np.float64)
                        w_sum = w.sum()
                        if w_sum > 0:
                            weights.append(w / w_sum)
                # Trim to requested count
                if len(weights) > n_weights:
                    # Select evenly spaced subset
                    indices = np.linspace(0, len(weights) - 1, n_weights, dtype=int)
                    weights = [weights[i] for i in indices]
            return np.array(weights)

        # Tchebycheff scalarization
        def _tchebycheff(f, w, ideal):
            """Compute Tchebycheff scalarized value."""
            return np.max(w * (f - ideal))

        # Initialize with diverse points using Tchebycheff
        n_weights = 20
        weights = _generate_weights(n_obj, n_weights)

        # First, get approximate ideal point by evaluating a few random points
        n_init = min(30, budget // 4)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = n_init

        # Estimate ideal point (minimum of each objective)
        ideal = np.min(F_init, axis=0)
        # Estimate nadir point (maximum of each objective)
        nadir = np.max(F_init, axis=0)
        range_f = nadir - ideal
        range_f = np.where(range_f < 1e-10, 1.0, range_f)

        # Initialize population at Tchebycheff weight locations
        pop_size = 24
        X = np.zeros((pop_size, dim))
        F_pop = np.zeros((pop_size, n_obj))

        # Assign each individual to a weight vector
        weight_assign = rng.choice(n_weights, size=pop_size)

        # Initialize each individual near its weight's preferred region
        for i in range(pop_size):
            w = weights[weight_assign[i]]
            # Start from a random point, then do a few local steps toward the weight direction
            x = lo + (hi - lo) * rng.random(dim)
            # Quick local optimization for this weight
            for step in range(3):
                if used >= budget:
                    break
                # Perturb
                delta = (hi - lo) * rng.normal(0, 0.1, dim)
                x_new = np.clip(x + delta, lo, hi)
                f_new = problem(x_new)
                used += 1
                # Accept if it improves the Tchebycheff value
                if _tchebycheff(f_new, w, ideal) < _tchebycheff(problem(x), w, ideal):
                    x = x_new
            X[i] = x
            F_pop[i] = problem(x)
            used += 1

        # Archive for non-dominated solutions
        archive_F = []
        for i in range(pop_size):
            if _is_non_dominated(archive_F, F_pop[i]):
                archive_F = _remove_dominated(archive_F, F_pop[i])
                archive_F.append(F_pop[i])

        # DE loop with Tchebycheff-based selection
        F_de = 0.8
        CR = 0.9

        # Track best Tchebycheff value per weight for convergence monitoring
        best_tv = np.full(n_weights, np.inf)
        for i in range(pop_size):
            w = weights[weight_assign[i]]
            tv = _tchebycheff(F_pop[i], w, ideal)
            if tv < best_tv[weight_assign[i]]:
                best_tv[weight_assign[i]] = tv

        stagnation_counter = 0
        max_stagnation = 15

        while used < budget and time.time() < deadline - 10:
            for i in range(pop_size):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Select parents using Tchebycheff tournament
                # Pick 3 random, choose the one with best Tchebycheff value for its weight
                idxs = list(range(pop_size))
                a_idx, b_idx, c_idx = rng.choice(idxs, size=3, replace=False)

                # Select best of the three by Tchebycheff value for their assigned weight
                def _tv_for(idx):
                    w = weights[weight_assign[idx]]
                    return _tchebycheff(F_pop[idx], w, ideal)

                candidates = [a_idx, b_idx, c_idx]
                a = min(candidates, key=_tv_for)

                # Pick two more distinct random individuals for mutation
                remaining = [j for j in idxs if j != i and j != a]
                if len(remaining) >= 2:
                    b, c = rng.choice(remaining, size=2, replace=False)
                else:
                    b, c = rng.choice(idxs, size=2, replace=False)
                    if b == c:
                        c = (c + 1) % pop_size
                    if b == i:
                        b = (b + 1) % pop_size
                    if c == i:
                        c = (c + 1) % pop_size

                # Create trial vector
                v = X[a] + F_de * (X[b] - X[c])

                # Binomial crossover
                u = X[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]

                # Clip to bounds
                u = np.clip(u, lo, hi)

                # Evaluate trial vector
                f_u = problem(u)
                used += 1

                # Selection based on Tchebycheff value for the assigned weight
                w_i = weights[weight_assign[i]]
                tv_u = _tchebycheff(f_u, w_i, ideal)
                tv_i = _tchebycheff(F_pop[i], w_i, ideal)

                if tv_u < tv_i:
                    X[i] = u
                    F_pop[i] = f_u
                    # Update best Tchebycheff value
                    if tv_u < best_tv[weight_assign[i]]:
                        best_tv[weight_assign[i]] = tv_u
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

                # Update archive
                if _is_non_dominated(archive_F, F_pop[i]):
                    archive_F = _remove_dominated(archive_F, F_pop[i])
                    archive_F.append(F_pop[i])

            # Check for stagnation and adapt
            if stagnation_counter > max_stagnation:
                # Perturb some individuals to escape local optima
                n_perturb = max(3, pop_size // 4)
                perturb_idx = rng.choice(pop_size, size=n_perturb, replace=False)
                for idx in perturb_idx:
                    if used >= budget:
                        break
                    delta = (hi - lo) * rng.normal(0, 0.3, dim)
                    X[idx] = np.clip(X[idx] + delta, lo, hi)
                    F_pop[idx] = problem(X[idx])
                    used += 1
                    # Update archive
                    if _is_non_dominated(archive_F, F_pop[idx]):
                        archive_F = _remove_dominated(archive_F, F_pop[idx])
                        archive_F.append(F_pop[idx])
                stagnation_counter = 0
                max_stagnation = int(max_stagnation * 1.5)

        # Local Refinement: Nelder-Mead on best archive points
        if used < budget - 10 and time.time() < deadline - 20:
            # Select top diverse points from archive
            if len(archive_F) > 0:
                sorted_indices = np.argsort(np.sum(archive_F, axis=1))
                candidates = []
                for idx in sorted_indices[:5]:
                    if len(candidates) >= 3:
                        break
                    # Check diversity
                    is_diverse = True
                    for c in candidates:
                        if np.linalg.norm(archive_F[idx] - c) < 0.1 * np.max(range_f):
                            is_diverse = False
                            break
                    if is_diverse:
                        candidates.append(archive_F[idx])

                # For each candidate, do local search using Tchebycheff with its nearest weight
                for f0 in candidates:
                    if used >= budget or time.time() > deadline - 5:
                        break

                    # Find nearest weight
                    dists = np.linalg.norm(weights - f0 / np.max(f0), axis=1)
                    w_local = weights[np.argmin(dists)]

                    # Simple local search
                    x_best = None
                    f_best = f0.copy()
                    tv_best = _tchebycheff(f0, w_local, ideal)

                    # Find the individual closest to this objective point
                    dists_x = np.linalg.norm(F_pop - f0, axis=1)
                    x_best = X[np.argmin(dists_x)].copy()

                    for step in range(8):
                        if used >= budget or time.time() > deadline - 5:
                            break
                        delta = (hi - lo) * rng.normal(0, 0.05, dim)
                        x_new = np.clip(x_best + delta, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        tv_new = _tchebycheff(f_new, w_local, ideal)
                        if tv_new < tv_best:
                            x_best = x_new
                            f_best = f_new
                            tv_best = tv_new

                    # Update archive
                    if _is_non_dominated(archive_F, f_best):
                        archive_F = _remove_dominated(archive_F, f_best)
                        archive_F.append(f_best)

        return archive_F

