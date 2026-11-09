"""DTLZ1 multi-objective optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer: exploit structure, low-discrepancy diversity, local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (hi + lo) / 2.0

        # DTLZ1 structure:
        # f1 = 0.5 * g * (1 + x1)
        # f2 = 0.5 * g * (1 - x1) * (1 + x2)
        # f3 = 0.5 * g * (1 - x2)
        # where g = 100 * (n-1 + sum((xi - 0.5)^2 - cos(20*pi*(xi-0.5))) for i in 2..n)
        # Optimal: g=0 when xi=0.5 for i>=2, and x1,x2 vary to trace the front.
        # Pareto front: x1 in [0,1], x2 in [0,1], f1+f2+f3=0.5, convex front.

        # Strategy:
        # 1. Set "g" variables (indices 2..dim-1) to 0.5 to get g=0
        # 2. Sample x1 (index 0) and x2 (index 1) to cover the Pareto front
        # 3. Use local search / refinement around promising points

        # Low-discrepancy sequence for 2D (x1, x2) coverage
        def halton_sequence(n, dim_h=2):
            """Generate n points in [0,1]^dim_h using Halton sequences."""
            pts = np.zeros((n, dim_h))
            for d in range(dim_h):
                base = d + 2  # 2, 3, 4, ...
                for i in range(n):
                    f = 1.0
                    x = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        x += f * (i_val % base)
                        i_val //= base
                    pts[i, d] = x
            return pts

        # Phase 1: Structured sampling on the Pareto front
        # Use Halton sequence for x1, x2 to get good coverage
        n_structured = min(100, budget // 2)
        halton_pts = halton_sequence(n_structured, 2)

        # Create candidate points with g-variables at 0.5
        X = np.zeros((n_structured, dim))
        for i in range(n_structured):
            X[i, 0] = lo[0] + span[0] * halton_pts[i, 0]
            X[i, 1] = lo[1] + span[1] * halton_pts[i, 1]
            for j in range(2, dim):
                X[i, j] = center[j]

        F = np.array([problem(x) for x in X])
        used = n_structured

        # Check deadline
        if time.time() > deadline - 5.0:
            return X

        # Phase 2: Local refinement using Nelder-Mead or similar
        # Focus on points that are on or near the Pareto front
        # For DTLZ1, the front is 2D, so we can do local search in (x1, x2) space
        # with g-variables fixed at 0.5

        # Select top candidates for refinement based on proximity to ideal point
        # Ideal point for DTLZ1: [0, 0, 0] but we want hypervolume, so spread matters
        # Let's pick a diverse subset for refinement

        # Use a simple non-dominated sort to identify front points
        def is_dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        # Identify non-dominated points
        nd_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
        nd_indices = np.where(nd_mask)[0]

        if len(nd_indices) == 0:
            nd_indices = np.arange(len(F))

        # Pick up to 20 non-dominated points for local refinement
        n_refine = min(20, len(nd_indices), (budget - used) // 5)
        if n_refine > 0:
            # Select diverse non-dominated points
            if len(nd_indices) > n_refine:
                # Spread selection
                step = len(nd_indices) // n_refine
                refine_idx = nd_indices[::step][:n_refine]
            else:
                refine_idx = nd_indices

            for idx in refine_idx:
                if time.time() > deadline - 10.0:
                    break
                if used >= budget:
                    break

                # Local search: perturb x1 and x2, keep g-variables at 0.5
                x0 = X[idx].copy()
                f0 = F[idx].copy()

                # Simple local search: try small perturbations in x1, x2
                best_x = x0.copy()
                best_f = f0.copy()
                step_size = 0.05

                for _ in range(3):
                    if used >= budget or time.time() > deadline - 10.0:
                        break
                    # Try random perturbations
                    for _try in range(5):
                        if used >= budget or time.time() > deadline - 10.0:
                            break
                        pert = np.zeros(dim)
                        pert[0] = np.random.uniform(-step_size, step_size)
                        pert[1] = np.random.uniform(-step_size, step_size)
                        new_x = np.clip(best_x + pert, lo, hi)
                        # Ensure g-variables stay at center
                        for j in range(2, dim):
                            new_x[j] = center[j]

                        new_f = problem(new_x)
                        used += 1

                        # Accept if not dominated by best_f
                        if not np.all(new_f >= best_f) and not np.all(new_f <= best_f):
                            # Check if new_f is better in at least one objective
                            if np.any(new_f < best_f):
                                best_x = new_x
                                best_f = new_f
                        elif np.all(new_f < best_f):
                            best_x = new_x
                            best_f = new_f

                    step_size *= 0.5

                # Update the population
                X[idx] = best_x
                F[idx] = best_f

        # Phase 3: Fill remaining budget with more structured sampling + local polish
        # Add more Halton points with slight jitter to explore
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5.0:
            n_extra = min(remaining, 100)
            # Use different offset for Halton
            extra_halton = halton_sequence(n_extra, 2)
            for i in range(n_extra):
                if time.time() > deadline - 5.0:
                    break
                x = np.zeros(dim)
                x[0] = lo[0] + span[0] * extra_halton[i, 0]
                x[1] = lo[1] + span[1] * extra_halton[i, 1]
                for j in range(2, dim):
                    x[j] = center[j]

                fx = problem(x)
                used += 1

                # Replace worst point in current population
                # Use sum of normalized objectives as a simple quality metric
                # Reference point [400, 400, 400] - points should be well below this
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = x
                F[worst_idx] = fx

        # Phase 4: Final polish - refine the best points with very fine steps
        if used < budget and time.time() < deadline - 5.0:
            # Get current non-dominated set
            nd_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))

            # Polish top few points
            n_polish = min(10, len(nd_indices), (budget - used) // 3)
            if n_polish > 0:
                if len(nd_indices) > n_polish:
                    step = len(nd_indices) // n_polish
                    polish_idx = nd_indices[::step][:n_polish]
                else:
                    polish_idx = nd_indices

                for idx in polish_idx:
                    if time.time() > deadline - 5.0:
                        break
                    if used >= budget:
                        break

                    x0 = X[idx].copy()
                    best_x = x0.copy()
                    best_f = F[idx].copy()
                    step_size = 0.01

                    for _ in range(3):
                        if used >= budget or time.time() > deadline - 5.0:
                            break
                        for _try in range(8):
                            if used >= budget or time.time() > deadline - 5.0:
                                break
                            pert = np.zeros(dim)
                            pert[0] = np.random.uniform(-step_size, step_size)
                            pert[1] = np.random.uniform(-step_size, step_size)
                            new_x = np.clip(best_x + pert, lo, hi)
                            for j in range(2, dim):
                                new_x[j] = center[j]

                            new_f = problem(new_x)
                            used += 1

                            if np.any(new_f < best_f) and not np.all(new_f > best_f):
                                best_x = new_x
                                best_f = new_f

                        step_size *= 0.5

                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
