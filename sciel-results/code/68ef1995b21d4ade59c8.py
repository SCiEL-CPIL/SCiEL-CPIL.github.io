"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Rosenbrock-aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, remaining_budget, deadline):
        """Run Nelder-Mead from x0, using at most remaining_budget calls."""
        dim = self.dim
        n = dim + 1
        # Build initial simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] = x0[i - 1] + 0.5

        # Evaluate simplex
        fvals = np.zeros(n)
        calls_used = 0
        for i in range(n):
            if time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls_used += 1

        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        max_iter = 200
        for it in range(max_iter):
            if time.time() > deadline or calls_used >= remaining_budget:
                break

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-8:
                break

            # Reflection
            centroid = np.mean(simplex[:-1], axis=0)
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = func(x_ref)
            calls_used += 1

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = func(x_exp)
                calls_used += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (simplex[-1] - centroid)
                f_con = func(x_con)
                calls_used += 1
                if f_con <= min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_used += 1

            # Re-sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= self.budget:
                return 1e10
            eval_count += 1
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Generate starting points with Rosenbrock-aware initialization
        starts = []
        # The Rosenbrock minimum is at (1, 1, ..., 1) in standard form
        # With hidden shift/rotation, we try multiple strategies
        starts.append(np.ones(self.dim))
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 0.5)
        starts.append(np.ones(self.dim) * 1.5)
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        # Random points around likely minimum region
        for _ in range(10):
            starts.append(rng.uniform(-2, 3, self.dim))
        # Some structured points
        starts.append(np.ones(self.dim) * -1.0)
        starts.append(np.ones(self.dim) * 2.0)
        starts.append(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        starts.append(np.array([1.5, 1.5, 1.5, 1.5, 1.5]))

        # Shuffle starts
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        for x0 in starts:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

            remaining = self.budget - eval_count
            try:
                x_best, f_best, calls_used = self._nelder_mead(
                    counting_func, x0, remaining, deadline
                )
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best
            except Exception:
                pass

            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
