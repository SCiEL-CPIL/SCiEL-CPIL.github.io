"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 with efficient search and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4 specific: x1 in [0, 11], x2..x10 in [-5, 5]
        # The Pareto front has f1 = x1, f2 = g(x2..x10) * h(x1)
        # where h(x1) = 1 - sqrt(x1/11) * exp(-x1 * exp(-4.0 * sum(xj^2) / 9.0 / dim))
        # g(x) = 1 + 10*(n-1) + sum((xj - 0.02)^2 * (1 + 10*sin(3*pi*xj))^2) for j=2..n

        # Strategy:
        # 1. Sample x1 uniformly in [0, 11] (this determines f1)
        # 2. For each x1, try to minimize f2 by optimizing x2..x10
        # 3. The minimum of f2 for a given x1 is approximately g_min * h(x1)
        #    where g_min occurs when sin terms are 0 and xj = 0.02

        # For ZDT4, the known Pareto front has:
        # f1 = x1, f2 = (1 + 10*9) * (1 - sqrt(x1/11) * exp(-x1 * exp(-0))) = 91 * (1 - sqrt(x1/11))
        # when all xj = 0.02 (so sin(3*pi*0.02) = sin(0.1884) ≈ 0.187, not zero)
        # Actually, the ideal point for g is when xj makes sin terms zero: xj = k/3
        # and xj - 0.02 = 0, i.e., xj = 0.02. But sin(3*pi*0.02) != 0.
        # The true minimum of g is more complex due to multimodality.

        # Let's use a smarter approach:
        # - Sample many points in x1 space
        # - For each x1, use a few evaluations to try to find good x2..x10
        # - Use the fact that x2..x10 should be near 0.02 or near values that make sin terms small

        # Better approach: Use a population-based method with exploitation
        # of the structure of ZDT4.

        # Key insight: For ZDT4, the best solutions have x2..x10 near 0.02
        # (or near other local minima of the multimodal function).
        # Let's sample x1 densely and for each x1, try x2..x10 = 0.02 as a baseline,
        # then do local search.

        # Phase 1: Dense sampling of x1 with x2..x10 = 0.02
        # This gives us the approximate Pareto front.

        # Number of x1 samples
        n_x1 = min(budget // 2, 100)
        x1_samples = np.linspace(0.0, 11.0, n_x1)

        # Baseline: x2..x10 = 0.02
        base_x = np.full(dim, 0.02)
        base_x[0] = 0.0  # will be set per sample

        F = []
        X = []
        used = 0

        # Evaluate baseline points
        for x1 in x1_samples:
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = x1
            f = problem(x)
            F.append(f)
            X.append(x)
            used += 1
            if used >= budget:
                break

        if used >= budget:
            return X

        # Phase 2: Local search around promising points
        # Convert to arrays
        F = np.array(F)
        X = np.array(X)

        # Find non-dominated points
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False

        # For remaining budget, do local search on non-dominated points
        # Sort by f1 to process in order
        order = np.argsort(F[:, 0])
        nondom_idx = []
        for i in order:
            if not is_dominated(F[i], F):
                nondom_idx.append(i)

        # Try to improve each non-dominated point
        for idx in nondom_idx:
            if used >= budget or time.time() > deadline:
                break

            x_best = X[idx].copy()
            f_best = F[idx].copy()

            # Local search: perturb x2..x10 slightly
            for _ in range(3):
                if used >= budget or time.time() > deadline:
                    break
                # Small perturbation in x2..x10
                perturbation = 0.05 * (hi[1:] - lo[1:]) * np.random.randn(dim - 1)
                x_try = x_best.copy()
                x_try[1:] = np.clip(x_try[1:] + perturbation, lo[1:], hi[1:])
                f_try = problem(x_try)
                used += 1

                # Check if this improves
                if not is_dominated(f_try, F):
                    # Replace or add
                    F = np.vstack([F, f_try])
                    X = np.vstack([X, x_try])
                    if np.sum(f_try) < np.sum(f_best):
                        x_best = x_try.copy()
                        f_best = f_try.copy()

        return X
# EVOLVE-BLOCK-END
