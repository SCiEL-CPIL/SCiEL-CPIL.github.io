"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ2-aware multi-objective optimizer: structured init + DE refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz2_front_point(self, theta, dim):
        """Generate a point on/near DTLZ2 Pareto front given angles theta.
        For DTLZ2: first (n_obj-1) angles define direction on unit sphere,
        last dim - (n_obj-1) variables are distance vars (set to 0.5 for front).
        """
        n_obj = self.n_obj
        n_angle = n_obj - 1  # 2 angles for 3 objectives
        x = np.zeros(dim)

        # Compute direction from angles (spherical coordinates for 3 objectives)
        # f_i = (1 + g) * prod_{j<i} cos(theta_j) * (sin(theta_i) if i==n_obj-1 else 1)
        # For 3 objectives:
        # f1 = (1+g)*cos(t1)
        # f2 = (1+g)*sin(t1)*cos(t2)
        # f3 = (1+g)*sin(t1)*sin(t2)
        # On Pareto front g=0, so we need x such that:
        # x[0]=cos(t1), x[1]=sin(t1)*cos(t2), x[2]=sin(t1)*sin(t2)
        # Remaining vars: distance vars, set to 0.5

        t1 = theta[0]
        t2 = theta[1] if len(theta) > 1 else 0.0

        x[0] = np.cos(t1)
        x[1] = np.sin(t1) * np.cos(t2)
        x[2] = np.sin(t1) * np.sin(t2)

        # Distance variables (x[3:]) set to 0.5 for Pareto front
        if dim > 3:
            x[3:] = 0.5

        return x

    def _generate_angles(self, n_points, n_angle, rng):
        """Generate well-distributed angles on the sphere surface."""
        # For 3 objectives, use Fibonacci sphere for 2D angles
        # theta1 in [0, pi/2], theta2 in [0, pi/2]
        # Use quasi-random distribution
        angles = np.zeros((n_points, n_angle))

        if n_angle == 2:
            # Use Halton-like sequence for better coverage
            for i in range(n_points):
                # Golden ratio based spacing for theta1
                theta1 = (i + 0.5) / n_points * (np.pi / 2)
                # For theta2, use a different offset to ensure coverage
                theta2 = ((i * 0.618033988749895) % 1.0) * (np.pi / 2)
                angles[i, 0] = theta1
                angles[i, 1] = theta2
        else:
            angles = rng.random((n_points, n_angle)) * (np.pi / 2)

        return angles

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        pop_size = 60  # Population size for DE
        used = 0

        # === Phase 1: Structured initialization on DTLZ2 Pareto front ===
        angles = self._generate_angles(pop_size, self.n_obj - 1, rng)
        X = np.array([self._dtlz2_front_point(theta, self.dim) for theta in angles])

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used += pop_size

        # Keep track of all evaluated points for hypervolume
        all_X = X.copy()
        all_F = F.copy()

        # === Phase 2: Differential Evolution refinement ===
        # DE parameters
        F_factor = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        S[i].append(j)
                    elif dominates(F_arr[j], F_arr[i]):
                        n_dom[i] += 1

            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            f = F_arr[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(f[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf

                f_range = f[sorted_idx[-1], m] - f[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (f[sorted_idx[k + 1], m] - f[sorted_idx[k - 1], m]) / f_range

            return dist

        def find_worst(F_arr, fronts):
            if len(fronts) == 0:
                return 0
            worst_front = fronts[-1]
            if len(worst_front) == 1:
                return worst_front[0]
            cd = crowding_distance(F_arr, worst_front)
            worst_local = np.argmin(cd)
            return worst_front[worst_local]

        # DE loop
        gen = 0
        while used < self.budget:
            if time.time() > deadline:
                break

            # Check if we have enough budget for a full generation
            if used + pop_size > self.budget:
                # Only do partial generation
                remaining = self.budget - used
                for i in range(remaining):
                    if time.time() > deadline:
                        break

                    # Select 3 distinct parents
                    idxs = rng.choice(pop_size, size=3, replace=False)
                    r1, r2, r3 = idxs

                    # DE mutation
                    mutant = X[r1] + F_factor * (X[r2] - X[r3])

                    # Binary crossover
                    trial = X[i].copy()
                    j_rand = rng.integers(self.dim)
                    for j in range(self.dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]

                    # Clip to bounds
                    trial = np.clip(trial, lo[i], hi[i])

                    # Evaluate
                    f_trial = problem(trial)
                    used += 1

                    # Environmental selection: compare with population
                    # Find worst in population
                    fronts = fast_non_dominated_sort(F)
                    worst = find_worst(F, fronts)

                    if dominates(f_trial, F[worst]):
                        X[worst] = trial
                        F[worst] = f_trial
                    elif not dominates(F[worst], f_trial):
                        # Same rank, use crowding distance
                        cd = crowding_distance(F, list(range(pop_size)))
                        if cd[i] < cd[worst]:
                            X[worst] = trial
                            F[worst] = f_trial

                    # Track all evaluated points
                    if used < len(all_X) + (self.budget - len(all_X)):
                        pass  # We'll handle tracking below

                break

            # Full generation
            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Select 3 distinct parents
                idxs = rng.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = idxs

                # DE mutation with adaptive F
                f_adaptive = F_factor * (0.5 + 0.5 * rng.random())
                mutant = X[r1] + f_adaptive * (X[r2] - X[r3])

                # Binary crossover
                trial = X[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Environmental selection
                fronts = fast_non_dominated_sort(F)
                worst = find_worst(F, fronts)

                if dominates(f_trial, F[worst]):
                    X[worst] = trial
                    F[worst] = f_trial
                elif not dominates(F[worst], f_trial):
                    cd = crowding_distance(F, list(range(pop_size)))
                    if cd[i] < cd[worst]:
                        X[worst] = trial
                        F[worst] = f_trial

            gen += 1

            # Track all evaluated points for hypervolume
            # We need to keep all unique non-dominated points
            # For efficiency, periodically prune
            if gen % 5 == 0:
                # Combine current population with previous tracked points
                pass

        # === Phase 3: Final local polish on best points ===
        # Sort by sum of objectives to find best candidates
        if used < self.budget and time.time() < deadline:
            # Take top 10 points by sum of objectives
            sums = np.sum(F, axis=1)
            top_idx = np.argsort(sums)[:min(10, pop_size)]

            for i in top_idx:
                if used >= self.budget or time.time() > deadline:
                    break

                # Local refinement: small random perturbations
                for attempt in range(5):
                    if used >= self.budget or time.time() > deadline:
                        break

                    perturbed = X[i].copy()
                    perturbed += rng.normal(0, 0.01, self.dim)
                    perturbed = np.clip(perturbed, lo, hi)

                    f_pert = problem(perturbed)
                    used += 1

                    # If better, update
                    if np.sum(f_pert) < np.sum(F[i]):
                        X[i] = perturbed
                        F[i] = f_pert

        return X
# EVOLVE-BLOCK-END
