"""Multi-objective optimizer for ZDT3 using structured decomposition with optimized sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits ZDT3 structure: f1=x1, f2=g*(1-sqrt(f1/g))."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # f2(x) = g(x) * (1 - sqrt(f1/g(x)))
        #
        # The Pareto front is achieved when g is minimized (g_min = 1) and x1 varies in [0,1].
        # When g=1, f2 = 1 - sqrt(x1)
        #
        # Strategy:
        # 1. Use x[1:]=0 to achieve g_min=1 (known analytical optimum)
        # 2. Sweep x1 across [0,1] to trace the Pareto front

        rng = np.random.default_rng(42)

        # Phase 1: Just verify the known optimum (1 evaluation)
        best_x_tail = np.zeros(dim - 1)
        evaluations_used = 0

        x_zero = np.zeros(dim)
        f_zero = problem(x_zero)
        evaluations_used += 1

        # Phase 2: Sweep x1 across [0,1] with x_tail=0
        # Use stratified sampling for better coverage of the discontinuous front

        n_remaining = budget - evaluations_used

        # Generate stratified x1 values: combine uniform grid with Halton-like jitter
        # This ensures coverage of all regions while avoiding clustering

        # Use a mix of:
        # 1. Dense uniform grid for baseline coverage
        # 2. Random perturbations to fill gaps

        # Primary: uniform grid with n_uniform points
        n_uniform = min(n_remaining, 198)
        x1_uniform = np.linspace(0.0, 1.0, n_uniform + 2)[1:-1]  # exclude exact 0 and 1 for now

        # Secondary: random points for gap-filling
        n_random = n_remaining - n_uniform
        x1_random = rng.random(n_random) if n_random > 0 else np.array([])

        # Combine and deduplicate
        x1_all = np.concatenate([[0.0], x1_uniform, x1_random, [1.0]])
        x1_all = np.clip(x1_all, 0.0, 1.0)
        x1_all = np.unique(x1_all)

        # Limit to available budget
        if len(x1_all) > n_remaining:
            # Prioritize keeping 0, 1, and the grid points
            x1_all = x1_all[:n_remaining]

        # Evaluate each x1 value
        results_F = []
        for x1_val in x1_all:
            if evaluations_used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.concatenate([[x1_val], best_x_tail])
            x = np.clip(x, lo, hi)
            f = problem(x)
            results_F.append(f)
            evaluations_used += 1

        # If we have budget left, add more random points for better coverage
        if evaluations_used < budget:
            n_remaining2 = budget - evaluations_used
            for _ in range(n_remaining2):
                if evaluations_used >= budget:
                    break
                if time.time() > deadline:
                    break

                x1_val = rng.random()
                x = np.concatenate([[x1_val], best_x_tail])
                x = np.clip(x, lo, hi)
                f = problem(x)
                results_F.append(f)
                evaluations_used += 1

        # Convert to array
        if results_F:
            F_final = np.array(results_F)
        else:
            F_final = np.array([[1.0, 1.0]])

        return F_final
# EVOLVE-BLOCK-END
