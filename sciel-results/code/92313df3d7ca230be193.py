"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Global-Local Hybrid Optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Sampling to find promising basins
        # We use a fraction of the budget for global exploration.
        # For dim=5 and Schwefel, a coarse grid or random sampling can find good starting points.
        # Let's use 10% of the budget for global search, capped at 50 calls to ensure we have enough for local.
        global_budget = min(50, budget // 10)
        if global_budget < 10:
            global_budget = 10
            
        best_x_global = None
        best_f_global = np.inf
        candidates = []
        
        # Use Halton sequence or simple random for diversity? 
        # Random is sufficient for 5D if we have enough samples.
        # Let's do stratified random sampling to cover the space better.
        
        # To keep it simple and fast, we'll just do random sampling but ensure we track the best.
        # Actually, for Schwefel, the global minimum is often far from 0.
        # Let's sample uniformly in [-5, 5]^n.
        
        calls_made = 0
        
        # We will generate 'global_budget' points.
        # We can batch them to reduce overhead, but func is a black box call.
        
        for _ in range(global_budget):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            # Sample in [-5, 5]
            x = np.random.uniform(-5, 5, n)
            f = func(x)
            calls_made += 1
            
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
                
            # Keep top 3 candidates for local refinement
            # Simple insertion sort into a small list
            if len(candidates) < 3:
                candidates.append((f, x.copy()))
                candidates.sort(key=lambda t: t[0])
            else:
                if f < candidates[-1][0]:
                    candidates[-1] = (f, x.copy())
                    candidates.sort(key=lambda t: t[0])

        # If no global calls were made (budget too small), fallback to single local search from 0
        if best_x_global is None:
            best_x_global = np.zeros(n)
            best_f_global = func(best_x_global)
            calls_made += 1
            candidates = [(best_f_global, best_x_global.copy())]

        # Phase 2: Local Refinement using (1+1)-ES for each candidate
        # We distribute the remaining budget among the candidates.
        # Since the best candidate is most likely to yield the best result, we give it more budget.
        # Strategy: 
        #   - Top candidate: 50% of remaining budget
        #   - Second: 30%
        #   - Third: 20%
        #   - If only 1 candidate, it gets 100%.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x_global, best_f_global

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Allocate budget
        num_cands = len(candidates)
        allocations = []
        if num_cands == 1:
            allocations = [remaining_budget]
        elif num_cands == 2:
            allocations = [int(remaining_budget * 0.6), int(remaining_budget * 0.4)]
        else:
            allocations = [int(remaining_budget * 0.5), int(remaining_budget * 0.3), int(remaining_budget * 0.2)]
        
        # Ensure allocations don't exceed remaining budget and are positive
        total_alloc = sum(allocations)
        if total_alloc > remaining_budget:
            # Scale down
            factor = remaining_budget / total_alloc
            allocations = [int(a * factor) for a in allocations]
            # Fix rounding error
            diff = remaining_budget - sum(allocations)
            if diff > 0:
                allocations[0] += diff
        
        current_best_x = best_x_global
        current_best_f = best_f_global

        for i, (f_init, x_init) in enumerate(candidates):
            if i >= len(allocations):
                break
            local_budget = allocations[i]
            if local_budget <= 0:
                break
                
            # Run (1+1)-ES starting from x_init
            # Initial step size: adaptive based on dimension
            sigma = 1.0 
            x = x_init.copy()
            f = f_init
            best_local_x = x.copy()
            best_local_f = f
            
            # (1+1)-ES loop
            # We use a simple step size adaptation: if improvement, increase slightly, else decrease.
            # Or standard (1+1)-ES with sigma adaptation.
            # For speed and robustness, let's use a simple hill-climbing with random restarts if stuck?
            # No, (1+1)-ES is better.
            
            # Parameters for (1+1)-ES
            # We will run until local_budget is exhausted or time is up.
            
            for _ in range(local_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Generate offspring
                # Clip sigma to avoid huge steps
                sigma = np.clip(sigma, 1e-6, 5.0)
                
                z = np.random.randn(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                calls_made += 1
                
                if f_y < f:
                    # Improvement
                    x = y
                    f = f_y
                    # Increase step size slightly (exploration)
                    sigma *= 1.05 
                    if f < best_local_f:
                        best_local_f = f
                        best_local_x = y.copy()
                else:
                    # No improvement
                    # Decrease step size (exploitation)
                    sigma *= 0.95
                    
                # Update global best
                if f < current_best_f:
                    current_best_f = f
                    current_best_x = x.copy()
                    
            # If we still have budget left and this wasn't the last candidate, 
            # we could potentially re-allocate, but the fixed allocation is simpler and safer.
            
        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
