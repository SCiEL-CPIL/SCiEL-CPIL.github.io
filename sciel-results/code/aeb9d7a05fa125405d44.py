"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Coarse-to-fine adaptive lattice search.
    1. Global coarse grid exploration to identify the best region.
    2. Local refinement using a fine grid around the best point.
    3. Final polish with a local (1+1)-ES style hill climb if budget allows.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        rng = np.random.default_rng(123)
        
        # State
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Global Exploration
        # Use a grid of size 3^dim or 4^dim depending on budget.
        # For dim=5, 3^5 = 243 points. 4^5 = 1024 points.
        # Budget is 500. Let's try a 3x3x3x3x3 grid first (243 calls).
        # If we have more budget, we can do 4x4x4x4x4 (1024 calls) but that exceeds budget.
        # So we stick to 3^dim for the global phase.
        
        grid_size = 3
        # Generate grid points
        # We want to cover [-5, 5]. 
        # Points: -5, 0, 5 for size 3.
        # To be more effective, maybe center around 0? 
        # The problem states minimizer is in [-4, 4]^5.
        # So a grid on [-4, 4] might be better? Or just [-5, 5].
        # Let's use [-5, 5] to be safe, but the peaks are likely well-distributed.
        
        # Actually, for f21, the peaks are quite spread out.
        # A 3-point grid per dimension might be too coarse if a peak is narrow.
        # But f21 peaks are not extremely narrow.
        
        # Let's use a Halton sequence or similar quasi-random for better coverage?
        # Or just a simple grid.
        # Let's try a 4-point grid on [-5, 5] -> 4^5 = 1024 points. Too many.
        # Let's try a 3-point grid on [-5, 5] -> 243 points.
        
        # Alternative: Use a small number of random starts + local search?
        # The inspiration says "coarse structured global exploration (Halton)".
        # Let's use a Halton sequence for the global phase. It provides better low-discrepancy coverage than a simple grid.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate N global candidates using Halton sequence
        # Map from [0,1]^d to [-5,5]^d
        n_global = 100  # Use 100 calls for global exploration
        global_candidates = []
        for i in range(1, n_global + 1):
            point = np.array([halton(i, base) for base in range(2, self.dim + 2)])
            # Map to [-5, 5]
            point = point * 10.0 - 5.0
            global_candidates.append(point)

        # Evaluate global candidates
        for p in global_candidates:
            if time.time() > deadline:
                break
            eval_point(p)

        # Phase 2: Local Refinement around the best point
        # Use a fine grid or local random search around best_x.
        # Let's use a local (1+1)-ES style search but with a fixed number of iterations.
        # Or a fine grid around the best point.
        
        # Let's do a fine grid around the best point.
        # Search region: best_x +/- 1.0 (adjustable)
        # Grid size: 3x3x3x3x3 = 243 points.
        # Total calls so far: ~100. Remaining: ~400.
        # We can afford a 3^5 grid (243) and then maybe another layer or local ES.
        
        # Let's try a local grid search.
        # Step size: 0.5
        # Range: [-1, 1] around best_x -> 5 points per dim -> 5^5 = 3125 points. Too many.
        # Range: [-0.5, 0.5] around best_x with step 0.25 -> 5 points per dim?
        # Let's use 3 points per dim in a box of size 1.0 (step 0.5).
        # Points: -0.5, 0, 0.5 relative to center.
        
        local_grid_size = 3
        local_step = 0.5
        local_candidates = []
        for i in range(local_grid_size):
            for j in range(local_grid_size):
                for k in range(local_grid_size):
                    for l in range(local_grid_size):
                        for m in range(local_grid_size):
                            offset = np.array([
                                (i - 1) * local_step,
                                (j - 1) * local_step,
                                (k - 1) * local_step,
                                (l - 1) * local_step,
                                (m - 1) * local_step
                            ])
                            local_candidates.append(best_x + offset)
        
        # Evaluate local candidates
        for p in local_candidates:
            if time.time() > deadline:
                break
            eval_point(p)

        # Phase 3: Final Polish with (1+1)-ES
        # Use remaining budget to run a (1+1)-ES starting from the new best_x.
        x = best_x.copy()
        fx = best_f
        sigma = 0.1  # Start with small sigma for polish
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            if sigma < 1e-4:
                break

        return x, fx
# EVOLVE-BLOCK-END
