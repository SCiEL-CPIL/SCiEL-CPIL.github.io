"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Diagonal (1+1)-ES with per-dimension step size adaptation for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_diagonal(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with diagonal step sizes and 1/5 success rule per dimension.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigmas = np.full(dim, sigma0)
        calls = 1
        
        # 1/5 success rule factors
        c_inc = 1.1
        c_dec = 0.9
        
        best_x = x.copy()
        best_f = fx
        
        # Track per-dimension success counts for adaptive sigma
        success_counts = np.zeros(dim)
        total_counts = np.zeros(dim)
        
        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break
                
            z = rng.standard_normal(dim)
            y = x + sigmas * z
            # Clip to box [-5, 5]
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Increase sigmas for dimensions where the step was positive
                # and decreased for negative (or vice versa based on improvement)
                # Simple approach: increase all sigmas
                sigmas *= c_inc
            else:
                # Failure
                sigmas *= c_dec
            
            # Prevent sigmas from becoming too small or too large
            sigmas = np.clip(sigmas, 1e-8, 10.0)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Strategy: Use multiple restarts with adaptive budget allocation
        # Start with a few broad explorations, then exploit the best region
        
        # Phase 1: Broad exploration with several short runs
        num_phase1 = 8
        calls_phase1 = max(20, budget // (num_phase1 * 2))
        
        starts = [np.zeros(dim)]
        for _ in range(num_phase1 - 1):
            x0 = rng.uniform(-4.5, 4.5, size=dim)
            starts.append(x0)
        
        sigma0_phase1 = 3.0
        
        for x0 in starts:
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            remaining = budget - total_calls
            max_calls = min(calls_phase1, remaining)
            if max_calls <= 1:
                break
                
            bx, bf, used = self._es_diagonal(func, rng, x0, sigma0_phase1, max_calls, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # Phase 2: Exploit the best region with more budget
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5.0:
            # Run a longer ES from the best point found
            sigma0_phase2 = 1.0
            bx, bf, used = self._es_diagonal(func, rng, best_x, sigma0_phase2, remaining, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
