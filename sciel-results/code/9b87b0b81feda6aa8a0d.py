"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with DTLZ7-specific priors."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ7 structure: x[0]=f1, x[1]=f2, x[2:]=y (controls f3)
        # Pareto front: y_i near 1 for all i gives f3 near 1
        # Initialize with y biased toward 1 (the front), x1,x2 spread
        def make_individual():
            x = np.empty(dim)
            x[0] = rng.random()  # f1
            x[1] = rng.random()  # f2
            # y variables: bias toward 1 (front) with some spread
            y = 1.0 - 0.5 * rng.random(dim - 2)  # in [0.5, 1.0]
            # Occasionally allow more exploration
            if rng.random() < 0.3:
                y = rng.random(dim - 2)
            x[2:] = y
            return np.clip(x, lo, hi)

        # Population size
        pop_size = min(40, max(10, self.budget // 5))

        # Initial population
        X = np.array([make_individual() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume (keep non-dominated archive)
        all_F = F.copy()
        all_X = X.copy()

        def fast_non_dominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for points at given indices."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf
            dist = np.zeros(len(indices))
            f = F[indices]
            for m in range(n_obj):
            # Sort by objective m
                order = np.argsort(f[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin = f[order[0], m]
                fmax = f[order[-1], m]
                if fmax - fmin < 1e-15:
                    continue
                for i in range(1, len(order) - 1):
                    dist[order[i]] += (f[order[i+1], m] - f[order[i-1], m]) / (fmax - fmin)
            return dist

        def dominates(a, b):
            """True if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated front
            nondom = fast_non_dominated(F)
            if len(nondom) == 0:
                nondom = np.array([0])

            # Select parent: prefer non-dominated, use tournament
            if rng.random() < 0.7 and len(nondom) > 0:
                parent_idx = nondom[rng.integers(len(nondom))]
            else:
                # Tournament selection among all
                t1 = rng.integers(pop_size)
                t2 = rng.integers(pop_size)
                if dominates(F[t1], F[t2]):
                    parent_idx = t1
                elif dominates(F[t2], F[t1]):
                    parent_idx = t2
                else:
                    parent_idx = t1 if rng.random() < 0.5 else t2

            parent = X[parent_idx]

            # Mutation: DTLZ7-aware
            # Keep y near 1 (the front region), perturb x1, x2 more freely
            child = parent.copy()
            # x1, x2: larger perturbation to explore f1, f2
            child[0] += 0.15 * span[0] * rng.standard_normal()
            child[1] += 0.15 * span[1] * rng.standard_normal()
            # y variables: small perturbation, bias toward 1
            y_noise = 0.05 * span[2:] * rng.standard_normal(dim - 2)
            child[2:] += y_noise
            # Pull y toward 1 (front)
            child[2:] += 0.1 * (1.0 - child[2:])
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            all_F = np.vstack([all_F, fc])
            all_X = np.vstack([all_X, child])

            # Replace worst in population (by sum of objectives, but prefer to keep diversity)
            # Use a simple strategy: replace a dominated point if possible, else worst
            dom_mask = np.array([any(dominates(F[i], F[j]) for i in range(pop_size) if i != j) for j in range(pop_size)])
            if np.any(dom_mask):
                dominated_idx = np.where(dom_mask)[0]
                worst = dominated_idx[np.argmax(np.sum(F[dominated_idx], axis=1))]
            else:
                worst = int(np.argmax(np.sum(F, axis=1)))

            X[worst] = child
            F[worst] = fc

        return all_X
# EVOLVE-BLOCK-END
