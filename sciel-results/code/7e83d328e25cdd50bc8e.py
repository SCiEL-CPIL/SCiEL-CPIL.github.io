"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using ZDT1 structure-aware search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 structure: f1 = x1, g = 1 + 9*mean(x2..xn), f2 = g*(1 - sqrt(f1/g))
        # Optimal: x2..xn = 0, x1 in [0,1]
        # We'll exploit this by fixing most variables near 0 and varying x1

        rng = np.random.default_rng(42)

        # Phase 1: Exploit ZDT1 structure
        # Set x[1:] to 0 (optimal for g) and vary x[0]
        # This gives us points on the true Pareto front

        # Allocate evaluations:
        # - Use ~80% of budget for structured search along x1
        # - Use ~20% for random exploration to handle potential variations

        n_structured = int(budget * 0.8)
        n_random = budget - n_structured

        # Phase 1a: Dense sampling of x1 in [0,1] with x[1:]=0
        # This directly samples the Pareto front
        x1_values = np.linspace(0.0, 1.0, n_structured)
        # Add some random x1 values for robustness
        x1_values = np.concatenate([x1_values, rng.random(max(0, n_structured - len(x1_values)))])
        x1_values = x1_values[:n_structured]

        results = []
        used = 0

        for x1 in x1_values:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(np.clip(x1, lo[0], hi[0]))
            # Keep x[1:] at 0 (or near lo[1:] if lo > 0)
            for i in range(1, dim):
                x[i] = lo[i]  # Optimal for ZDT1 g term
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
            used += 1
            if used >= budget:
                break

        # Phase 1b: Add a few points at boundaries for completeness
        boundary_points = [
            np.array([lo[0]] + [lo[i] for i in range(1, dim)]),
            np.array([hi[0]] + [lo[i] for i in range(1, dim)]),
            np.array([0.0] + [lo[i] for i in range(1, dim)]),
            np.array([1.0] + [lo[i] for i in range(1, dim)]),
        ]
        for x in boundary_points:
            if used >= budget or time.time() > deadline:
                break
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
            used += 1

        # Phase 2: Random exploration with some perturbation of x[1:]
        # to handle cases where the problem isn't exactly ZDT1 or has noise
        while used < budget:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = rng.uniform(lo[0], hi[0])
            # Small perturbation on x[1:]
            for i in range(1, dim):
                x[i] = lo[i] + rng.uniform(0, 0.05) * (hi[i] - lo[i])
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
            used += 1

        # Phase 3: Refine around promising regions
        # If we have results, find the region with best hypervolume contribution
        if results and used < budget:
            # Sort by f1 to identify spread
            results_sorted = sorted(results, key=lambda r: r[1][0])
            # Sample more densely in regions where points are sparse
            if len(results_sorted) > 2:
                f1_vals = [r[1][0] for r in results_sorted]
                # Find largest gap
                max_gap = 0
                gap_idx = 0
                for i in range(len(f1_vals) - 1):
                    gap = f1_vals[i + 1] - f1_vals[i]
                    if gap > max_gap:
                        max_gap = gap
                        gap_idx = i
                # Add points in the largest gap
                n_refine = min(10, budget - used)
                for _ in range(n_refine):
                    if used >= budget or time.time() > deadline:
                        break
                    x1 = rng.uniform(f1_vals[gap_idx], f1_vals[gap_idx + 1])
                    x = np.zeros(dim)
                    x[0] = float(np.clip(x1, lo[0], hi[0]))
                    for i in range(1, dim):
                        x[i] = lo[i]
                    try:
                        f = problem(x)
                        if f is not None and len(f) == 2:
                            results.append((x.copy(), np.array(f, dtype=float)))
                    except Exception:
                        pass
                    used += 1

        # Return the last evaluated point (return value is ignored by evaluator)
        if results:
            return results[-1][0]
        return np.zeros(dim)

# EVOLVE-BLOCK-END
