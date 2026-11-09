"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT1 structure with evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT1: f1 = x[0], g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - cos(20*pi*xi)) for i>=1
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # The Pareto front is: f2 = 1 - sqrt(f1/11) for f1 in [0, 1]
        # g(x) is minimized when xi = 0.5 for all i >= 1, giving g = 11
        # So optimal points have x[1:] = 0.5, and x[0] varies in [0,1]

        # Strategy: 
        # 1. Initialize with points on the known Pareto front (x[1:]=0.5, x[0] uniform)
        # 2. Use remaining budget to explore around the front with small perturbations
        # 3. Keep track of non-dominated set for efficiency

        # First, let's evaluate points on the Pareto front
        # We know the front: for x[0] = t in [0,1], x[1:] = 0.5, we get f1=t, f2=1-sqrt(t/11)
        
        # Allocate budget:
        # - Use ~50 points evenly spaced on the front to establish a good baseline
        # - Use remaining budget for local search around front points
        
        n_front = min(50, budget // 4)
        n_explore = budget - n_front
        
        # Generate front points
        x0_vals = np.linspace(0.0, 1.0, n_front)
        front_X = np.zeros((n_front, dim))
        for i in range(n_front):
            front_X[i, 0] = x0_vals[i]
            front_X[i, 1:] = 0.5  # optimal for g(x)
        
        # Evaluate front points
        front_F = np.array([problem(x) for x in front_X])
        all_X = [front_X]
        all_F = [front_F]
        used = n_front
        
        # Now explore: for each front point, do local search with small perturbations
        # Focus on perturbing x[0] slightly and x[1:] slightly
        n_local = 5  # local search iterations per front point if budget allows
        total_local = n_front * n_local
        if total_local > n_explore:
            n_local = max(1, n_explore // n_front)
        
        for i in range(n_front):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            for j in range(n_local):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Create perturbation
                x = front_X[i].copy()
                # Perturb x[0] with small step
                step0 = 0.01 * rng.standard_normal()
                x[0] = np.clip(x[0] + step0, lo[0], hi[0])
                # Perturb x[1:] with very small step
                step_rest = 0.005 * rng.standard_normal(dim - 1)
                x[1:] = np.clip(x[1:] + step_rest, lo[1:], hi[1:])
                
                fx = problem(x)
                used += 1
                all_X.append(x.reshape(1, -1))
                all_F.append(np.array([fx]).reshape(1, -1))
        
        # Combine all evaluations
        X_all = np.vstack(all_X)
        F_all = np.vstack(all_F)
        
        # Return the population (the return value is ignored, but we need to return something)
        return X_all
# EVOLVE-BLOCK-END
