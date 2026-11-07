"""Multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DEMO-style multi-objective optimizer for DTLZ1 with local polishing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a larger population for better diversity
        pop_size = min(40, max(20, budget // 5))

        # Deterministic initialization: spread points evenly in the first dim-1 dimensions
        # For DTLZ1, the Pareto front is defined by the first dim-1 variables
        # We initialize by sampling the simplex uniformly
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # Sample points on the simplex for the first dim-1 dimensions
            # Use Dirichlet distribution for uniform sampling on simplex
            if dim > 1:
                # Generate random numbers and normalize
                r = np.random.random(dim - 1)
                r = r / r.sum()
                X[i, :dim-1] = r
            # Last dimension is free, set to 0.5 initially
            X[i, dim-1] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final selection
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_d = np.zeros(n, dtype=int)  # domination counts
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                if n_d[p] == 0:
                    fronts[0].append(p)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            # Remove the last empty front
            if not fronts[-1]:
                fronts.pop()
            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to indices in F."""
            if len(indices) <= 2:
                return {i: np.inf for i in indices}
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                if f_max - f_min < 1e-12:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[k]] += (F[sorted_idx[k + 1]][m] - F[sorted_idx[k - 1]][m]) / (f_max - f_min)
            return dist

        # Main evolution loop using DE (Differential Evolution)
        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents: pick from first front (non-dominated)
            parent_pool = fronts[0]
            if len(parent_pool) < 3:
                # Include second front if available
                if len(fronts) > 1:
                    parent_pool = parent_pool + fronts[1]
            if len(parent_pool) < 3:
                parent_pool = list(range(len(F)))

            # DE mutation: x_trial = x_rand1 + F * (x_rand2 - x_rand3)
            idx = np.random.randint(len(F))
            x_base = X[idx].copy()
            
            # Select 3 different parents
            p1 = np.random.choice(parent_pool, replace=False, size=min(3, len(parent_pool)))
            if len(p1) < 3:
                # If not enough, pad with random
                while len(p1) < 3:
                    p1 = np.append(p1, np.random.randint(len(F)))
            
            p1, p2, p3 = p1[0], p1[1], p1[2]
            if p1 == idx or p2 == idx or p3 == idx:
                continue

            # DE mutation
            F_factor = 0.5 + 0.5 * np.random.random()
            x_trial = X[p1] + F_factor * (X[p2] - X[p3])

            # Crossover (binomial)
            CR = 0.5
            for j in range(dim):
                if np.random.random() < CR or j == np.random.randint(dim):
                    x_trial[j] = x_trial[j]
                # else keep base

            # Mutation: small random perturbation
            if np.random.random() < 0.5:
                x_trial += 0.1 * np.random.randn(dim)

            x_trial = np.clip(x_trial, lo, hi)
            fc = problem(x_trial)
            used += 1

            # Selection: keep the better of x_trial and x_base
            # Use non-domination and crowding distance
            # Simple approach: if x_trial dominates x_base, replace
            # If x_base dominates x_trial, keep x_base
            # Otherwise, keep the one with better crowding distance (approximated by random for speed)
            
            # Check domination
            trial_dominates_base = np.all(fc <= F[idx]) and np.any(fc < F[idx])
            base_dominates_trial = np.all(F[idx] <= fc) and np.any(F[idx] < fc)

            if trial_dominates_base or (not base_dominates_trial and np.random.random() < 0.5):
                X[idx] = x_trial
                F[idx] = fc
                all_X.append(X.copy())
                all_F.append(F.copy())

        # Final local polishing pass: refine points near the Pareto front
        # Use coordinate-wise line search on the best non-dominated points
        if used < budget and time.time() < deadline:
            # Get non-dominated points
            fronts = fast_non_dominated_sort(F)
            nd_indices = fronts[0]
            
            # For each non-dominated point, do a small local search
            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                x_curr = X[idx].copy()
                f_curr = F[idx].copy()
                
                # Try small perturbations in each dimension
                for dim_idx in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    # Try moving in positive and negative directions
                    for direction in [-1, 1]:
                        step = 0.01
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        if np.allclose(x_trial, x_curr):
                            continue
                        
                        f_trial = problem(x_trial)
                        used += 1
                        
                        # Check if better
                        if np.all(f_trial <= f_curr) and np.any(f_trial < f_curr):
                            x_curr = x_trial
                            f_curr = f_trial
                            X[idx] = x_curr
                            F[idx] = f_curr
                            all_X.append(X.copy())
                            all_F.append(F.copy())

        return X
# EVOLVE-BLOCK-END
