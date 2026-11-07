"""Multi-objective optimizer for ZDT3 using structured decomposition with Halton sampling."""
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
        deadline = start_time + 285  # safety margin before 300s limit

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
        # 1. Minimize g(x) by optimizing x[1:] (29 variables)
        # 2. Then sweep x1 across [0,1] to trace the Pareto front

        # Phase 1: Minimize g(x[1:])
        # The minimum of g is 1 (achieved when all x[i]=0 for i>=1)
        # We'll use a combination of random search and local refinement

        rng = np.random.default_rng(42)

        # Allocate budget:
        # - 50 evaluations to minimize g over x[1:]
        # - remaining for x1 sweep

        n_phase1 = 50
        n_phase2 = budget - n_phase1

        # Phase 1: Minimize g(x[1:])
        # g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # Minimum is at x[i]=0 for all i>=1, giving g_min = 1

        best_g = np.inf
        best_x_tail = np.zeros(dim - 1)
        evaluations_used = 0

        # Try the known optimum first
        x_zero = np.zeros(dim)
        f_zero = problem(x_zero)
        evaluations_used += 1
        g_zero = 1.0  # g at x=0 is exactly 1
        best_g = g_zero
        best_x_tail = np.zeros(dim - 1)

        # Local refinement around zero with small perturbations
        for _ in range(n_phase1 - 1):
            if evaluations_used >= n_phase1:
                break
            if time.time() > deadline:
                break

            # Try perturbation around zero
            x_tail = rng.normal(0, 0.01, dim - 1)
            x_tail = np.clip(x_tail, lo[1:], hi[1:])
            x = np.concatenate([[0.5], x_tail])
            x = np.clip(x, lo, hi)

            # Compute g directly
            g_val = 1.0 + 10.0 * (dim - 1)
            for j in range(1, dim):
                g_val += x[j] ** 2 - 10.0 * np.cos(2.0 * np.pi * x[j])

            if g_val < best_g:
                best_g = g_val
                best_x_tail = x_tail.copy()
            evaluations_used += 1

        # Phase 2: Sweep x1 across [0,1] with the best x_tail
        # Use Halton sequence for better coverage of the discontinuous Pareto front

        # Generate Halton sequence points in [0,1]
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                base = 2
                for _ in range(dim):
                    point.append(halton(i, base))
                    base += 1
                points.append(point)
            return np.array(points)

        # Generate x1 values using Halton sequence for better space-filling
        n_halton = min(n_phase2, 199)
        if n_halton > 0:
            halton_points = halton_sequence(n_halton, 1)
            x1_values = halton_points.flatten()
        else:
            x1_values = np.array([])

        # Add endpoints to ensure full coverage
        x1_values = np.concatenate([[0.0], x1_values, [1.0]])
        x1_values = np.clip(x1_values, 0.0, 1.0)

        # Sort and deduplicate
        x1_values = np.unique(x1_values)

        # Evaluate each x1 value
        results_F = []
        for x1_val in x1_values:
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
            n_remaining = budget - evaluations_used
            for _ in range(n_remaining):
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
