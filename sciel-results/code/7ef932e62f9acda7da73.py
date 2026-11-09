"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search then CMA-ES style local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse search with larger steps to find the basin
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 3.0  # large initial step for coarse search
        for i in range(1, min(budget, 100)):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            if fy < best_f:
                best_f, best_x = fy, y.copy()

        # Phase 2: Fine-tuning with smaller steps
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        for i in range(100, budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.05
            else:
                sigma *= 0.95
            if fy < best_f:
                best_f, best_x = fy, y.copy()

        # Phase 3: Very fine local refinement
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        for i in range(budget, budget * 2):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.02
            else:
                sigma *= 0.98
            if fy < best_f:
                best_f, best_x = fy, y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
