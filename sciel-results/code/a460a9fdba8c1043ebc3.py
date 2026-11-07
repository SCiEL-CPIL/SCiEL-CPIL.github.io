"""Rewritten optimizer using Particle Swarm Optimization (PSO)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """PSO continuous optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # PSO Parameters
        # Population size needs to be small enough to fit in budget
        # Let's use a swarm size of 20. 500 / 20 = 25 iterations.
        # Or smaller swarm for more iterations? 
        # Standard PSO often works well with N=10-30.
        # Let's try N=10 to allow 50 iterations. More iterations might be better for convergence.
        # Or N=20 for 25 iterations.
        # Let's go with N=10 for finer resolution in time.
        N = 10
        
        # Velocity coefficients
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social
        w_start = 0.9
        w_end = 0.4
        
        # Initialize swarm
        # Random uniform in [-5, 5]
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within run? 
        # Task says "hidden seeded shift... instances apply a hidden seeded shift". 
        # Our RNG should probably be random or fixed? 
        # Using a fixed seed inside the optimizer makes it deterministic for the same instance? 
        # But instances are different. 
        # It is safer to use a random seed or default. 
        # However, for debugging, fixed is good. 
        # The evaluator creates Optimizer(budget=500, dim=5) and calls it.
        # If I use a fixed seed, the search path is the same for all instances? 
        # That might be bad if the shift is different. 
        # Better to use a random seed based on time or just default rng.
        rng = np.random.default_rng()

        X = rng.uniform(-5, 5, size=(N, n))
        V = rng.uniform(-1, 1, size=(N, n)) # Initial velocity small
        
        # Evaluate initial swarm
        F = np.array([func(x) for x in X])
        calls = N
        
        # Update personal bests
        P_best = X.copy()
        F_P_best = F.copy()
        
        # Update global best
        g_idx = np.argmin(F_P_best)
        G_best = P_best[g_idx].copy()
        F_G_best = F_P_best[g_idx]
        
        # Total iterations
        total_iters = budget // N
        # If there's a remainder, we can do one last partial iteration or just ignore
        # The contract says "More than 500 calls... makes the whole result invalid".
        # So we must stop exactly at 500 or less.
        # We will do `total_iters` full iterations.
        # If budget is not divisible by N, we can use the remaining calls for a few extra particles or just stop.
        # Let's just do full iterations.
        
        for it in range(total_iters):
            # Update inertia weight
            w = w_start - (w_start - w_end) * (it / max(1, total_iters - 1))
            
            # Update velocity
            # V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
            r1 = rng.random((N, n))
            r2 = rng.random((N, n))
            
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
            
            # Update position
            X = X + V
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate new positions
            # We have N calls here.
            F = np.array([func(x) for x in X])
            calls += N
            
            # Update personal bests
            improved = F < F_P_best
            P_best[improved] = X[improved]
            F_P_best[improved] = F[improved]
            
            # Update global best
            local_g_idx = np.argmin(F_P_best)
            if F_P_best[local_g_idx] < F_G_best:
                F_G_best = F_P_best[local_g_idx]
                G_best = P_best[local_g_idx].copy()
        
        # Use remaining budget if any (budget % N)
        remaining = budget - calls
        if remaining > 0:
            # Use remaining calls to refine the global best area
            # Simple local search around G_best
            # Generate `remaining` random points near G_best
            # Or just evaluate G_best again? No, we already have it.
            # Let's do a small random perturbation search.
            for _ in range(remaining):
                # Perturb G_best
                perturbation = rng.normal(0, 0.5, size=n)
                x_new = G_best + perturbation
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < F_G_best:
                    F_G_best = f_new
                    G_best = x_new.copy()
                    
        return G_best, F_G_best
# EVOLVE-BLOCK-END
