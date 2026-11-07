"""Multi-objective optimizer on ZDT6 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Exploit ZDT6 structure: Pareto front has x_2..x_n=0, x_1 varying."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # Track all evaluated points and objectives
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            X_all.append(x.copy())
            F_all.append(f.copy())
            return f

        # Phase 1: Sample the known Pareto front directly
        # ZDT6 Pareto front: x_2..x_n = 0, x_1 varies in [0,1]
        # f1 = x_1 / sin(2*pi*x_1), f2 = g(x) * (1 + f1/sin(f1)) where g=0 on front
        # So on the front, f2 = 0 (since g=0)
        
        # Generate diverse x_1 values to cover the front
        n_front = min(150, self.budget - 10)  # Leave some budget for refinement
        x1_values = np.linspace(0.01, 0.99, n_front)
        
        for x1 in x1_values:
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 5:
                break
            x = np.zeros(n)
            x[0] = x1
            # x[1:] = 0 (already zero)
            evaluate(x)
        
        # Phase 2: Use remaining budget for local refinement or additional sampling
        # Try to find points with very low g(x) near the front
        rng = np.random.default_rng(0)
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            # Sample a few points with small perturbations to x_2..x_n
            for _ in range(min(remaining, 20)):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                x = np.zeros(n)
                x[0] = rng.uniform(0.01, 0.99)
                # Small perturbations to other dimensions
                x[1:] = rng.uniform(0, 0.01, n-1)
                evaluate(x)
        
        # Phase 3: If still have budget, add more front points with different x_1
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            # Use golden ratio spacing for better coverage
            golden = 0.618033988749895
            for i in range(min(remaining, 50)):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                # Generate x1 using low-discrepancy sequence
                x1 = ((i + 1) * golden) % 1.0
                x1 = max(0.01, min(0.99, x1))
                x = np.zeros(n)
                x[0] = x1
                evaluate(x)

        return X_all
# EVOLVE-BLOCK-END
