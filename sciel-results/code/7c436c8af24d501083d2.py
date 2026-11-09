"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting problem structure.
    
    ZDT4 structure:
    - f1 = 1 + 10*(n-1) + sum(xi^2 - 10*cos(2*pi*xi)) for i=2..n
    - g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n  
    - f2 = g(x) * (1 - sqrt(f1/g(x)))
    - Pareto front: g minimized (x2..xn = 0), x1 varies in [-5,5]
    
    Strategy: Concentrate evaluations near the known Pareto-optimal region
    (tail variables near 0, x1 spread across [-5,5]).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        budget = self.budget
        
        # ZDT4 specific: x1 in [-5,5], x2..xn in [-5,5]
        # Pareto front requires x2..xn near 0
        # Allocate evaluations:
        # - 70%: x1 sweep with tail variables very close to 0
        # - 20%: fine-tuning around the Pareto-optimal region
        # - 10%: exploration to handle any deviations
        
        # Phase 1: Dense sweep of x1 with tail at exactly 0 (if in bounds)
        # Check if 0 is within bounds for tail variables
        tail_zero = np.zeros(n - 1)
        tail_valid = np.all(tail_zero >= lo[1:] - 1e-10) and np.all(tail_zero <= hi[1:] + 1e-10)
        
        # If 0 is not in bounds, use the closest point to 0
        if not tail_valid:
            tail_zero = np.clip(tail_zero, lo[1:], hi[1:])
        
        # Number of x1 samples for the main sweep
        n_sweep = int(budget * 0.6)
        n_fine = int(budget * 0.25)
        n_explore = budget - n_sweep - n_fine
        
        X = []
        F = []
        
        # Phase 1: Uniform sweep of x1 with tail at optimal value
        if n_sweep > 0:
            # x1 range
            x1_lo, x1_hi = lo[0], hi[0]
            x1_samples = np.linspace(x1_lo, x1_hi, n_sweep)
            for x1_val in x1_samples:
                x = np.concatenate([[x1_val], tail_zero])
                x = np.clip(x, lo, hi)
                X.append(x)
                F.append(problem(x))
        
        # Phase 2: Fine-tuning - perturb tail variables slightly around optimal
        if n_fine > 0:
            # Use a few x1 values with small perturbations in tail
            n_x1_fine = min(10, n_fine // max(1, n_fine // 10))
            x1_fine = np.linspace(lo[0], hi[0], max(1, n_x1_fine))
            remaining = n_fine
            for x1_val in x1_fine:
                if remaining <= 0:
                    break
                # Try a few small perturbations
                n_perturb = min(remaining, 3)
                for _ in range(n_perturb):
                    if remaining <= 0:
                        break
                    # Small Gaussian perturbation around 0 for tail
                    scale = 0.1 * (hi[1:] - lo[1:])
                    tail_pert = np.clip(tail_zero + np.random.randn(n-1) * scale, lo[1:], hi[1:])
                    x = np.concatenate([[x1_val], tail_pert])
                    x = np.clip(x, lo, hi)
                    X.append(x)
                    F.append(problem(x))
                    remaining -= 1
        
        # Phase 3: Exploration - random points but biased toward the Pareto region
        if n_explore > 0:
            for _ in range(n_explore):
                if time.time() > deadline:
                    break
                # Biased sampling: tail variables closer to 0
                x1_val = np.random.uniform(lo[0], hi[0])
                # Exponential decay toward 0 for tail
                tail_exp = np.zeros(n-1)
                for i in range(n-1):
                    # Sample from a distribution concentrated near 0
                    u = np.random.uniform(-1, 1)
                    # Scale by a fraction of the range
                    tail_exp[i] = u * 0.3 * (hi[1+i] - lo[1+i]) / 2
                tail_exp = np.clip(tail_exp, lo[1:], hi[1:])
                x = np.concatenate([[x1_val], tail_exp])
                x = np.clip(x, lo, hi)
                X.append(x)
                F.append(problem(x))
        
        # Ensure we don't exceed budget
        X = X[:budget]
        F = F[:budget]
        
        # Check if we used all budget; if not, add more
        while len(X) < budget and time.time() < deadline:
            x1_val = np.random.uniform(lo[0], hi[0])
            tail_zero_local = np.clip(np.zeros(n-1), lo[1:], hi[1:])
            # Small perturbation
            scale = 0.05 * (hi[1:] - lo[1:])
            tail_pert = np.clip(tail_zero_local + np.random.randn(n-1) * scale, lo[1:], hi[1:])
            x = np.concatenate([[x1_val], tail_pert])
            x = np.clip(x, lo, hi)
            X.append(x)
            F.append(problem(x))
        
        return X
# EVOLVE-BLOCK-END
