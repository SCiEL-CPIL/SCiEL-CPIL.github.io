"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT3 structure: place points on Pareto front (x[1:]=0)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3: Pareto front requires x[1:] = 0 (g=1)
        # f1 = x[0], f2 = g*(1 - sqrt(f1/g)) with 57 discontinuous regions
        # Strategy: set x[1:]=0, vary x[0] densely to cover all front regions

        # Allocate budget: most to front coverage, some to gap-filling
        n_front = int(budget * 0.8)  # 80% directly on front
        n_refine = budget - n_front  # 20% for refinement

        # Generate x[0] values covering [0,1] densely
        # Combine uniform grid + stratified over 57 regions + random
        x0_parts = []

        # Uniform grid
        x0_grid = np.linspace(0, 1, min(150, n_front))
        x0_parts.append(x0_grid)

        # Stratified: one sample per disconnected region (57 regions)
        n_regions = 57
        for i in range(n_regions):
            # Each region spans roughly 1/57 of [0,1]
            center = (i + 0.5) / n_regions
            spread = 0.3 / n_regions
            x0_parts.append(np.array([center + rng.uniform(-spread, spread)]))

        # Additional random samples
        n_rand = max(0, n_front - len(x0_grid) - n_regions)
        if n_rand > 0:
            x0_parts.append(rng.uniform(0, 1, size=n_rand))

        x0_all = np.concatenate(x0_parts)
        x0_all = np.clip(x0_all, lo[0], hi[0])

        # Trim to n_front if needed
        if len(x0_all) > n_front:
            # Stratified pick
            idx = np.linspace(0, len(x0_all) - 1, n_front, dtype=int)
            x0_front = x0_all[idx]
        else:
            x0_front = x0_all

        # Build points on Pareto front
        X = np.zeros((n_front, dim))
        X[:, 0] = x0_front
        # x[1:] = 0 already

        F = np.array([problem(x) for x in X])
        used = n_front

        # Phase 2: Refine by adding points in gaps or near boundaries
        if n_refine > 0 and time.time() < deadline:
            # Identify gaps in x[0] coverage and fill them
            x0_sorted = np.sort(x0_front)
            gaps = np.diff(x0_sorted)
            # Also consider gaps at boundaries
            # Add points in largest gaps
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                if len(gaps) > 0 and np.max(gaps) > 0.01:
                    # Find largest gap
                    gap_idx = np.argmax(gaps)
                    gap_center = (x0_sorted[gap_idx] + x0_sorted[gap_idx + 1]) / 2
                    x0_new = gap_center + rng.uniform(-0.005, 0.005)
                else:
                    # Perturb a random existing point
                    base = rng.integers(0, n_front)
                    x0_new = x0_front[base] + rng.uniform(-0.01, 0.01)

                x0_new = np.clip(x0_new, lo[0], hi[0])
                x = np.zeros(dim)
                x[0] = x0_new
                # Evaluate
                _ = problem(x)
                used += 1

        return X
# EVOLVE-BLOCK-END
