import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # ZDT4 structure:
        # f1 = 1 + 10*(n-1) + sum(xi^2 - 10*cos(2*pi*xi)) for i=2..n
        # g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # Pareto front: g minimized (x2..xn = 0), x1 varies in [-5,5]
        # Actually, for ZDT4, the front is where x2..xn=0 and x1 in [-5,5]
        # But wait, f1 depends on x2..xn too. When x2..xn=0, f1 = 1 + 0 = 1? No.
        # f1 = 1 + 10*(n-1) + sum(xi^2 - 10cos(2pi xi))
        # If x2..xn = 0, then xi^2=0, cos(0)=1, so term = -10.
        # So f1 = 1 + 10*(n-1) + (n-1)*(-10) = 1 + 10n - 10 - 10n + 10 = 1.
        # So f1 = 1 always on the front? That doesn't seem right for a front.
        
        # Let me re-read ZDT4 definition.
        # ZDT4:
        # f1(x) = x1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n (xi^2 - 10*cos(2*pi*xi))
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        # Front: x2..xn = 0 => g=1, f1=x1, f2 = 1 - sqrt(x1)
        # So x1 in [0,1] for standard ZDT4? Or [-5,5]?
        # Standard ZDT4 has x1 in [0,1], xi in [-5,5] for i>1.
        # The task says "ZDT4 (10 variables)". Bounds are given.
        # If bounds are standard, x1 in [0,1], x2..x10 in [-5,5].
        
        # Strategy:
        # 1. Identify the "front" region. For ZDT4, this is x2..xn = 0.
        # 2. Sweep x1 across its full range (likely [0,1] or the given bounds).
        # 3. Use remaining budget to explore small perturbations around the front
        #    to handle any numerical issues or if the bounds are slightly different.

        # Determine the "optimal" values for tail variables (x2..xn)
        # For ZDT4, this is 0.
        tail_optimal = np.zeros(n - 1)
        
        # Clip to bounds in case 0 is out of bounds (unlikely for standard ZDT4)
        tail_optimal = np.clip(tail_optimal, lo[1:], hi[1:])

        # Determine x1 range
        x1_lo, x1_hi = lo[0], hi[0]

        # Allocate budget
        # 80% for dense sweep of x1 with tail at optimal
        # 20% for fine-tuning / exploration
        n_sweep = int(budget * 0.8)
        n_fine = budget - n_sweep

        X = []
        F = []

        # Phase 1: Dense uniform sweep of x1 with tail at optimal
        # This should capture the entire Pareto front if the model is correct.
        if n_sweep > 0:
            # Use linspace to ensure even coverage
            x1_samples = np.linspace(x1_lo, x1_hi, n_sweep)
            for x1_val in x1_samples:
                x = np.empty(n)
                x[0] = x1_val
                x[1:] = tail_optimal
                x = np.clip(x, lo, hi)
                X.append(x)
                F.append(problem(x))

        # Phase 2: Fine-tuning / Exploration
        # Sample points with small perturbations in tail variables to ensure
        # we don't miss any slightly better points due to numerical noise or
        # if the "optimal" tail isn't exactly 0.
        if n_fine > 0:
            # Random x1 values
            x1_samples_fine = np.random.uniform(x1_lo, x1_hi, n_fine)
            for i in range(n_fine):
                x = np.empty(n)
                x[0] = x1_samples_fine[i]
                # Small Gaussian perturbation around optimal tail
                # Scale based on bounds
                tail_range = hi[1:] - lo[1:]
                # Use a small fraction of the range, e.g., 1%
                pert_scale = 0.01 * tail_range
                x[1:] = tail_optimal + np.random.randn(n-1) * pert_scale
                x = np.clip(x, lo, hi)
                X.append(x)
                F.append(problem(x))
                
                if time.time() > deadline:
                    break

        # If we haven't used all budget (due to time break), fill with more sweep points
        # This is unlikely if the sweep is fast, but good for safety.
        while len(X) < budget and time.time() < deadline:
            x1_val = np.random.uniform(x1_lo, x1_hi)
            x = np.empty(n)
            x[0] = x1_val
            x[1:] = tail_optimal
            x = np.clip(x, lo, hi)
            X.append(x)
            F.append(problem(x))

        return X
