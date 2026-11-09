"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # For DTLZ1, the Pareto front is characterized by:
        # x[0] and x[1] determine the position on the front via:
        #   f0 = 0.5 * x[0] * (1 + g)
        #   f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        #   f2 = 0.5 * x[0] * x[1] * (1 + g)
        # where g depends on x[2:]. To minimize g, we need to find optimal x[2:].
        #
        # Strategy:
        # 1. Probe multiple candidate values for x[2:] to find the g-minimizing configuration
        # 2. Sample points on the Pareto front by varying x[0], x[1]
        # 3. Use remaining budget for local refinement

        # Step 1: Find good g-minimizing configuration for x[2:]
        # Try several candidate configurations for x[2:]
        candidates_x2 = []
        # All zeros
        candidates_x2.append(np.zeros(dim - 2))
        # All ones
        candidates_x2.append(np.ones(dim - 2))
        # All 0.5
        candidates_x2.append(np.full(dim - 2, 0.5))
        # Random samples
        rng = np.random.default_rng(42)
        for _ in range(10):
            candidates_x2.append(rng.random(dim - 2))

        # Evaluate each candidate to find the one with smallest objective sum
        best_g_val = np.inf
        best_x2 = None
        for x2 in candidates_x2:
            x_test = np.concatenate([[0.5, 0.5], x2])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_val = np.sum(f)
            if g_val < best_g_val:
                best_g_val = g_val
                best_x2 = x2.copy()

        # Refine x[2:] with a few local searches
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x2_perturbed = best_x2 + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
            x2_perturbed = np.clip(x2_perturbed, lo[2:], hi[2:])
            x_test = np.concatenate([[0.5, 0.5], x2_perturbed])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_val = np.sum(f)
            if g_val < best_g_val:
                best_g_val = g_val
                best_x2 = x2_perturbed.copy()

        # Step 2: Sample points on the Pareto front
        # The front is parameterized by x[0] in [0,1] and x[1] in [0,1]
        # with x[2:] fixed at best_x2
        # f0 = 0.5 * x[0] * (1 + g)
        # f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        # f2 = 0.5 * x[0] * x[1] * (1 + g)
        #
        # We need to estimate g. From our best probe:
        # g ≈ (sum(f) - 0.5 * x[0] * (1 + g)) ... this is circular.
        # Instead, let's just sample x[0] and x[1] and evaluate.

        # Generate a grid of (x0, x1) pairs
        n_samples = min(100, self.budget - 20)  # leave some budget for refinement
        x0_vals = np.linspace(0.01, 0.99, int(np.sqrt(n_samples)))
        x1_vals = np.linspace(0.01, 0.99, int(np.sqrt(n_samples)))

        evaluated = 0
        recorded_F = []

        for x0 in x0_vals:
            if evaluated >= self.budget or time.time() > deadline:
                break
            for x1 in x1_vals:
                if evaluated >= self.budget or time.time() > deadline:
                    break
                x = np.concatenate([[x0, x1], best_x2])
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated += 1
                recorded_F.append(f)

        # Step 3: Use remaining budget for local refinement around promising points
        remaining = self.budget - evaluated
        if remaining > 0 and recorded_F:
            F_arr = np.array(recorded_F)
            # Find non-dominated points
            dominated = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if dominated[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break

            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) > 0:
                # Refine around non-dominated points
                for idx in non_dom_idx[:min(10, remaining)]:
                    if evaluated >= self.budget or time.time() > deadline:
                        break
                    # Try small perturbations
                    x_base = np.concatenate([[x0_vals[idx // len(x1_vals)], x1_vals[idx % len(x1_vals)]], best_x2])
                    for _ in range(2):
                        if evaluated >= self.budget or time.time() > deadline:
                            break
                        x = x_base + 0.05 * (hi - lo) * rng.standard_normal(dim)
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        evaluated += 1
                        recorded_F.append(f)

        return recorded_F
# EVOLVE-BLOCK-END
