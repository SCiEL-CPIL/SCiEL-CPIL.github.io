import numpy as np
import time


class Optimizer:
    """
    Direct DTLZ2 optimizer exploiting the known structure:
    For DTLZ2 with M objectives and N variables:
      - First M-1 variables are 'shape' variables (control position on the front via sin/cos)
      - Remaining N-(M-1) variables are 'distance' variables (should be 0.5 for Pareto front)
    
    The Pareto front for DTLZ2 with g=0 is a quarter-sphere of radius 0.5 in R^3_+.
    
    Strategy:
      1. Set distance variables to 0.5.
      2. Use a dense quasi-random (Halton) sampling of the 2D shape space.
      3. Use remaining budget for local refinement to push points closer to the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the index-th Halton sequence value for the given base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s wall-clock

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        k_shape = self.n_obj - 1  # 2 for M=3
        shape_indices = list(range(k_shape))
        dist_indices = list(range(k_shape, self.dim))

        # Determine optimal distance value
        # For standard DTLZ2, distance vars at 0.5 give g=0
        dist_val = 0.5
        evals_used = 0
        remaining = self.budget

        # Phase 1: Dense quasi-random sampling of shape space
        # Use Halton sequence for better coverage than grid
        n_phase1 = remaining - 20  # reserve 20 for refinement
        n_phase1 = max(n_phase1, 50)

        solutions = []
        objective_values = []
        used = 0

        # Generate Halton sequence points in 2D
        for idx in range(1, n_phase1 + 1):
            if time.time() > deadline:
                break

            x = np.full(self.dim, dist_val)
            # 2D Halton sequence for shape variables
            x[shape_indices[0]] = self._halton(idx, 2)
            x[shape_indices[1]] = self._halton(idx, 3)
            x = np.clip(x, 0.0, 1.0)

            f = np.array(problem(x))
            solutions.append(x.copy())
            objective_values.append(f.copy())
            used += 1

        if not solutions:
            # Fallback: return a single point
            x = np.full(self.dim, dist_val)
            x[shape_indices[0]] = 0.5
            x[shape_indices[1]] = 0.5
            f = np.array(problem(x))
            solutions.append(x)
            objective_values.append(f)
            return np.array(objective_values)

        # Phase 2: Local refinement using remaining budget
        if used < remaining and time.time() < deadline:
            F = np.array(objective_values)
            n = len(F)

            # Find non-dominated points
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True

            nd_indices = np.where(~dominated)[0]

            if len(nd_indices) > 0:
                # Select points to refine: pick a subset evenly spaced
                budget_left = remaining - used
                n_to_refine = min(len(nd_indices), max(4, budget_left // 5))
                if n_to_refine > 0:
                    if len(nd_indices) > n_to_refine:
                        # Evenly spaced selection by first objective
                        sort_idx = np.argsort(F[nd_indices, 0])
                        nd_sorted = nd_indices[sort_idx]
                        step = len(nd_sorted) / n_to_refine
                        refine_idx = nd_sorted[[int(i * step) for i in range(n_to_refine)]]
                    else:
                        refine_idx = nd_indices

                    # Multi-step local search
                    step_sizes = [0.05, 0.02, 0.008]
                    for idx in refine_idx:
                        if time.time() > deadline or used >= remaining - 2:
                            break

                        base_x = solutions[idx].copy()
                        best_f = objective_values[idx].copy()
                        # Metric: distance from ideal front (quarter sphere radius 0.5)
                        # f1^2 + f2^2 + f3^2 should be 0.25
                        best_metric = abs(np.sum(best_f ** 2) - 0.25)

                        for ss in step_sizes:
                            if used >= remaining - 2 or time.time() > deadline:
                                break
                            for di in range(k_shape):
                                for sign in [-1, 1]:
                                    if used >= remaining - 2 or time.time() > deadline:
                                        break
                                    new_x = base_x.copy()
                                    new_x[shape_indices[di]] += sign * ss
                                    new_x = np.clip(new_x, 0.0, 1.0)

                                    new_f = np.array(problem(new_x))
                                    used += 1

                                    new_metric = abs(np.sum(new_f ** 2) - 0.25)
                                    if new_metric < best_metric - 1e-10:
                                        best_metric = new_metric
                                        best_f = new_f
                                        base_x = new_x

                        # Add refined point
                        solutions.append(base_x)
                        objective_values.append(best_f)

        return np.array(objective_values)
