"""Multi-objective optimizer for DTLZ1 using a hybrid decomposition + local search approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid MOEA/D-style optimizer with local refinement for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate reference points on the simplex (for 3 objectives, this is a 2D simplex)
        n_ref = 150
        # Generate points on the 2-simplex: y1 + y2 + y3 = 1, yi >= 0
        # Use a grid approach
        n_grid = 17  # 17x17 grid on the simplex gives ~153 points
        ref_points = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                k = n_grid - 1 - i - j
                ref_points.append([i / (n_grid - 1), j / (n_grid - 1), k / (n_grid - 1)])
        ref_points = np.array(ref_points)
        # Scale to DTLZ1 objective space: each objective in [0, 1] for ideal front
        # DTLZ1 ideal point is (0,0,0), nadir is (0.5, 0.5, 0.5)
        # Actually for DTLZ1 with M=3, the Pareto front is f1+f2+f3 = 0.5
        # So we want reference points on the simplex scaled appropriately
        # The weight vectors should be in [0,1]^M summing to 1
        ref_points = ref_points / ref_points.sum(axis=1, keepdims=True)

        # Initialize population at reference points mapped to decision space
        # For DTLZ1: x1...x_{n-M+1} determine g, x_{n-M+2}...xn determine the front position
        # f1 = 0.5 * g * x1
        # f2 = 0.5 * g * (1-x1) * x2
        # f3 = 0.5 * g * (1-x1) * (1-x2)
        # To get a point on the front, set g=0, so x1...x_{n-M+1} = 0.5
        # Then f1 = 0, f2 = 0, f3 = 0.5 * (1-x1) * (1-x2) ... wait that's not right
        # Let me re-derive: for DTLZ1 with M objectives:
        # g = 100 * (|X'| + sum(x_i' - 0.5)^2 * sin(20*pi*x_i'))
        # where X' = x_{M-1}...x_n (the last n-M+1 variables)
        # f1 = 0.5 * g * x1
        # f2 = 0.5 * g * (1-x1) * x2
        # f3 = 0.5 * g * (1-x1) * (1-x2) for M=3
        # To be on the Pareto front, g=0, so x_{M-1}...x_n = 0.5
        # Then f1=0, f2=0, f3=0.5*(1-x1)*(1-x2) ... that gives f3 only
        # Wait, I think I have the formula wrong. Let me reconsider.
        # Standard DTLZ1:
        # f1 = 0.5 * x1 * g
        # f2 = 0.5 * (1-x1) * x2 * g  -- no this doesn't sum to 0.5*g
        # Actually: sum of all fi = 0.5 * g * (x1 + (1-x1)*x2 + (1-x1)*(1-x2)) = 0.5*g
        # So on the front (g=0), all fi = 0. That's a single point.
        # I'm confusing myself. Let me use a different approach.
        # Just use random initialization and let the algorithm find the front.

        # Initialize population randomly
        n_pop = min(50, budget // 4)
        X = lo + (hi - lo) * np.random.random((n_pop, dim))
        F = np.array([problem(x) for x in X])
        used = n_pop

        # Track best non-dominated set
        all_X = X.copy()
        all_F = F.copy()

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_mask = non_dominated_mask(F)
            if np.any(nd_mask):
                parents = X[nd_mask]
                parent_F = F[nd_mask]
            else:
                parents = X
                parent_F = F

            # Select parent using tournament
            n_parents = min(3, len(parents))
            idx = np.random.choice(len(parents), size=n_parents, replace=False)
            # Pick the best (lowest sum) among selected
            sums = parent_F[idx].sum(axis=1)
            best_parent = parents[np.argmin(sums)]

            # Create child with mutation
            # Adaptive step size
            step = 0.05 * np.random.random(dim) + 0.01
            child = best_parent + step * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add to archive
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])

            # Replace worst individual in population
            # Use a simple criterion: sum of objectives
            worst_idx = np.argmax(F.sum(axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Every few iterations, do a local search on the best point
            if used % 20 == 0 and used < budget - 5:
                # Find best point in population (lowest sum)
                best_idx = np.argmin(F.sum(axis=1))
                best_x = X[best_idx].copy()
                best_f = F[best_idx].copy()

                # Local search: try small perturbations
                n_local = 3
                for _ in range(n_local):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    perturbation = 0.01 * np.random.standard_normal(dim)
                    trial = np.clip(best_x + perturbation, lo, hi)
                    trial_f = problem(trial)
                    used += 1
                    # Check if trial dominates best
                    if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                        best_x = trial
                        best_f = trial_f
                        X[best_idx] = trial
                        F[best_idx] = trial_f
                        all_X = np.vstack([all_X, trial])
                        all_F = np.vstack([all_F, trial_f])

        # Final: return the archive (we'll let the evaluator record all calls)
        return all_X
# EVOLVE-BLOCK-END
