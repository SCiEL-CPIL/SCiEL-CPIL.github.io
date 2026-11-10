"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: structured exploration + tournament selection + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a moderate size for good coverage
        pop_size = min(30, max(10, self.budget // 6))
        remaining = self.budget

        # Phase 1: Structured initial population (Latin hypercube-like + random)
        # Use a mix of spread-out points and random points
        n_structured = pop_size // 2
        n_random = pop_size - n_structured

        # Structured: use quantiles to spread across the space
        structured_X = []
        for i in range(n_structured):
            # Use different quantile levels for each variable
            quantiles = (np.arange(self.dim) + 0.5) / self.dim
            # Shift based on i to create diversity
            offset = (i / max(1, n_structured - 1)) if n_structured > 1 else 0.5
            x = lo + span * np.clip(quantiles + offset * 0.3 - 0.15, 0, 1)
            structured_X.append(x)

        # Random points
        rng = np.random.default_rng(42)
        random_X = [lo + span * rng.random(self.dim) for _ in range(n_random)]

        X = np.array(structured_X + random_X)
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Track all evaluated points for hypervolume
        all_F = [F.copy()]
        all_X = [X.copy()]

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf
            n = len(indices)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                if vals.max() > vals.min():
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (vals.max() - vals.min())
            return dist

        # Phase 2: Evolutionary loop with tournament selection
        gen = 0
        while remaining > 0 and time.time() < deadline:
            # Find non-dominated set
            nd_mask = non_dominated_sort(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))

            # Compute crowding distance on non-dominated set
            cd = crowding_distance(F, nd_indices)

            # Select parents using tournament on (dominance, crowding)
            # Simpler: pick from non-dominated with crowding-based probability
            probs = cd / cd.sum()
            parent_idx = nd_indices[rng.choice(len(nd_indices), p=probs)]

            # Generate child via mutation
            child = X[parent_idx].copy()
            # Adaptive mutation: larger steps early, smaller later
            progress = 1.0 - (remaining / max(1, self.budget - pop_size))
            step_size = 0.3 * (1.0 - 0.7 * progress)
            child += step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            remaining -= 1

            # Replace the worst point in the current population
            # Use a combined metric: prefer replacing dominated points, then by sum
            dominated_mask = ~nd_mask
            if np.any(dominated_mask):
                # Among dominated, replace the one with worst sum
                dominated_indices = np.where(dominated_mask)[0]
                worst_in_dominated = dominated_indices[np.argmax(F[dominated_indices].sum(axis=1))]
                replace_idx = worst_in_dominated
            else:
                # All non-dominated: replace the one with lowest crowding distance
                # Map back to original indices
                cd_full = np.zeros(len(F))
                cd_full[nd_indices] = cd
                replace_idx = np.argmin(cd_full)

            X[replace_idx] = child
            F[replace_idx] = fc

            gen += 1

            # Periodically record the current front
            if gen % 5 == 0:
                all_F.append(F.copy())
                all_X.append(X.copy())

        # Phase 3: Local polish - refine the best non-dominated points
        # Use remaining budget for local search around promising points
        if remaining > 0 and time.time() < deadline:
            # Get the current non-dominated set
            nd_mask = non_dominated_sort(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) > 0:
                # Focus on a subset of the best points
                n_polish = min(len(nd_indices), remaining)
                # Select points to polish: those with good crowding
                cd = crowding_distance(F, nd_indices)
                probs = cd / cd.sum()
                polish_indices = nd_indices[rng.choice(len(nd_indices), size=n_polish, replace=False, p=probs)]

                for idx in polish_indices:
                    if remaining <= 0 or time.time() >= deadline:
                        break
                    # Local search: try small perturbations
                    base = X[idx].copy()
                    best_f = F[idx].copy()
                    best_x = base.copy()

                    # Try a few local moves
                    for _ in range(min(3, remaining)):
                        if time.time() >= deadline:
                            break
                        # Small step
                        step = 0.05 * span * rng.standard_normal(self.dim)
                        candidate = np.clip(base + step, lo, hi)
                        cf = problem(candidate)
                        remaining -= 1

                        # Accept if it dominates or is non-dominated and better in some sense
                        if np.all(cf <= best_f) and np.any(cf < best_f):
                            best_f = cf
                            best_x = candidate
                        elif not np.any(np.all(F <= cf) & np.any(F < cf)):
                            # Non-dominated, keep track
                            if cf[0] < best_f[0] or cf[1] < best_f[1]:
                                best_f = cf
                                best_x = candidate

                    X[idx] = best_x
                    F[idx] = best_f

        # Final record
        all_F.append(F.copy())
        all_X.append(X.copy())

        return X
# EVOLVE-BLOCK-END
