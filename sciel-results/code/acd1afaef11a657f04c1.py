"""Rewritten program for WFG7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using adaptive refinement along the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        scale = hi - lo

        # WFG7 has a known Pareto front. The front lies in the plane
        # where f1 = 1 - sqrt(f2), f3 = 2*f2 (approximately, for the standard WFG7).
        # Actually, for WFG7 with 12 variables, the Pareto front is:
        # f1 = 1 - sqrt(f2), f3 = 2*f2, with f2 in [0, 1]
        # But the actual bounds and scaling may differ. Let's use a parametric sampling.

        # Strategy: 
        # 1. Sample the known analytic Pareto front densely to establish initial spread.
        # 2. Use remaining budget for local line searches along tangent directions.

        # First, let's establish the Pareto front by sampling.
        # WFG7 Pareto front: for t in [0, 1], the ideal point is [1, 0, 0] and nadir is [0, 1, 2]
        # The front is f1 = 1 - sqrt(f2), f3 = 2*f2
        # So parametrize by f2 = t^2, t in [0,1]: f1 = 1 - t, f2 = t^2, f3 = 2*t^2

        # But we need to find the actual x values that produce these. 
        # Instead, let's do a hybrid: random sampling + greedy refinement.

        # Start with a set of diverse points
        n_init = min(20, self.budget // 5)
        used = 0

        # Generate initial population with good spread
        X = []
        F = []

        # Use a grid-like sampling in objective space, then map back
        # Since we can't invert WFG7 easily, use random but stratified
        rng = np.random.default_rng(42)

        # Initial stratified random sampling
        for i in range(n_init):
            if time.time() > deadline:
                break
            # Use Latin hypercube-like sampling
            t = (i + 0.5) / n_init
            x = mid + scale * (2 * rng.random(self.dim) - 1) * 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Now refine: for each point, try small perturbations in different directions
        # and keep the best non-dominated improvements

        # Define a set of search directions
        n_dirs = 6
        # Random directions + coordinate axes
        dirs = []
        for d in range(self.dim):
            e = np.zeros(self.dim)
            e[d] = 1.0
            dirs.append(e)
            e[d] = -1.0
            dirs.append(e)
        # Add some random directions
        for _ in range(n_dirs):
            d = rng.standard_normal(self.dim)
            d /= np.linalg.norm(d)
            dirs.append(d)

        # Adaptive step sizes
        base_step = 0.1

        # Refinement loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # For each point, try to improve it
            for i in range(len(X)):
                if used >= self.budget or time.time() > deadline:
                    break

                best_x = X[i]
                best_f = F[i]
                best_improved = False

                for d in dirs:
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Try a few step sizes
                    for step in [base_step, base_step * 0.5, base_step * 2]:
                        if used >= self.budget or time.time() > deadline:
                            break

                        new_x = best_x + step * scale * d
                        new_x = np.clip(new_x, lo, hi)
                        new_f = problem(new_x)
                        used += 1

                        # Check if new_f dominates best_f
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x
                            best_f = new_f
                            best_improved = True

                if best_improved:
                    X[i] = best_x
                    F[i] = best_f

            # Occasionally add new random points to escape local optima
            if used < self.budget - 5 and len(X) < 40:
                if time.time() > deadline:
                    break
                x = lo + scale * rng.random(self.dim)
                f = problem(x)
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                used += 1

        return X
# EVOLVE-BLOCK-END
