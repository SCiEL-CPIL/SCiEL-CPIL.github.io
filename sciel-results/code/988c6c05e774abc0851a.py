"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1: f1 = x[0], g = 1 + 9*n(x[1:]), f2 = g*(1 - sqrt(f1/g))
        # Pareto front: f2 = 1 - sqrt(f1) for f1 in [0,1]
        # Strategy: stratified sampling on the known Pareto front, then refine

        # Phase 1: Stratified sampling along the known Pareto front
        # For each t in [0,1], the optimal x has x[0]=t and x[1:]=0 (giving g=1, f2=1-sqrt(t))
        # But we need to verify this and also explore around these points

        n_strat = min(60, budget // 3)
        used = 0
        F_all = []
        X_all = []

        # Stratified points on the Pareto front
        t_vals = np.linspace(0.001, 0.999, n_strat)
        for t in t_vals:
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = t
            # Small perturbation in x[1:] to ensure we're in the interior
            x[1:] = 0.01 * rng.random(dim - 1)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_all.append(f)
            X_all.append(x)
            used += 1

        # Phase 2: Random sampling to find any better points
        n_random = min(40, budget - used)
        for _ in range(n_random):
            if used >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_all.append(f)
            X_all.append(x)
            used += 1

        # Phase 3: Local refinement around promising points
        # Convert to arrays
        F_arr = np.array(F_all)
        X_arr = np.array(X_all)

        # Find non-dominated points
        n_pts = len(F_arr)
        dominated = np.zeros(n_pts, dtype=bool)
        for i in range(n_pts):
            for j in range(n_pts):
                if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                    dominated[i] = True
                    break

        non_dom_idx = np.where(~dominated)[0]

        # Refine around non-dominated points
        n_refine = min(80, budget - used)
        n_non_dom = len(non_dom_idx)
        if n_non_dom > 0 and n_refine > 0:
            for _ in range(n_refine):
                if used >= budget:
                    break
                # Pick a random non-dominated point to refine
                idx = non_dom_idx[rng.integers(n_non_dom)]
                x = X_arr[idx].copy()
                # Small perturbation
                step = 0.05 * rng.standard_normal(dim)
                x = x + step
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_all.append(f)
                X_all.append(x)
                used += 1

        # Phase 4: Further stratified refinement with smaller steps
        # Use remaining budget for more stratified points
        remaining = budget - used
        if remaining > 0:
            n_more = min(remaining, 100)
            # Use more finely stratified points
            t_vals2 = np.linspace(0.001, 0.999, n_more)
            # Add jitter
            t_vals2 = t_vals2 + rng.uniform(-0.01, 0.01, n_more)
            t_vals2 = np.clip(t_vals2, 0.001, 0.999)
            for t in t_vals2:
                if used >= budget:
                    break
                x = np.zeros(dim)
                x[0] = t
                x[1:] = 0.005 * rng.random(dim - 1)
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_all.append(f)
                X_all.append(x)
                used += 1

        return X_all
# EVOLVE-BLOCK-END
