"""Optimizer for BBOB f15 (Rastrigin rotated) with multi-start and adaptive ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Broad exploration with multiple random starts
        # Use a subset of budget for exploration
        n_starts = 20
        evals_per_start = max(10, (self.budget // n_starts) // 2)
        total_budget_used = 0

        def run_es(start_x, start_f, max_evals, sigma_init=1.0):
            nonlocal best_x, best_f, total_budget_used
            x = start_x.copy()
            fx = start_f
            sigma = sigma_init
            no_improve = 0
            patience = 30

            for _ in range(max_evals):
                if total_budget_used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_budget_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 5.0)
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if no_improve >= patience:
                    break

            return x, fx

        # Generate initial diverse starts
        starts = []
        for i in range(n_starts):
            if time.time() > deadline:
                break
            # Mix of uniform and centered starts
            if i < n_starts // 2:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                x0 = rng.uniform(-2, 2, self.dim)
            fx0 = func(x0)
            total_budget_used += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            starts.append((x0, fx0))

        # Sort starts by fitness and run ES on best ones
        starts.sort(key=lambda t: t[1])

        # Run ES on the top starts, allocating more budget to better starts
        for i, (x0, fx0) in enumerate(starts):
            if total_budget_used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Allocate decreasing budget to worse starts
            remaining = self.budget - total_budget_used
            if remaining <= 0:
                break

            # First few starts get more budget
            if i < 5:
                alloc = min(remaining, max(20, remaining // 3))
            elif i < 10:
                alloc = min(remaining, max(10, remaining // 5))
            else:
                alloc = min(remaining, 5)

            if alloc < 5:
                break

            run_es(x0, fx0, alloc, sigma_init=max(0.5, 2.0 / (i + 1)))

        # Phase 2: Final refinement on the best point found
        if best_x is not None and total_budget_used < self.budget and time.time() < deadline:
            remaining = self.budget - total_budget_used
            if remaining > 10:
                # Small sigma for fine-tuning
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    total_budget_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.1, 0.5)
                    else:
                        sigma *= 0.95
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
