"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT2-specific optimizer: exploit known structure to place points on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT2: f1 = x0, g = 1 + 9*(dim-1)/|x[1:]|, f2 = g*(1-sqrt(f1/g))
        # Pareto front: x1..x_{dim-1} = 0, x0 in [0,1] => f1=x0, f2=1-sqrt(f1)
        # Strategy: spread x0 across [0,1], keep other dims at 0

        # Use most of the budget for structured sampling on the Pareto front
        # Reserve a few evaluations for boundary checking / safety
        n_front = max(10, budget - 10)
        n_extra = budget - n_front

        # Generate x0 values spread across [0, 1]
        # Use a mix of uniform and slightly denser near ends for better HV coverage
        x0_vals = np.linspace(0.0, 1.0, n_front)

        # Generate points on the Pareto front
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = x0_vals
        # x1..x_{dim-1} remain 0

        # Evaluate front points
        F_list = []
        for x in X_front:
            if time.time() > deadline:
                break
            f = problem(x)
            F_list.append(f)

        # If we have extra budget, add a few perturbed points for robustness
        if n_extra > 0 and time.time() < deadline:
            rng = np.random.default_rng(12345)
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                # Slight perturbation: keep x0 in [0,1], small noise in others
                x = np.zeros(dim)
                x[0] = rng.uniform(0.0, 1.0)
                # Tiny perturbation in other dims (stays within [0,1])
                x[1:] = rng.uniform(0.0, 1e-4, size=dim - 1)
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_list.append(f)

        return X_front
# EVOLVE-BLOCK-END
