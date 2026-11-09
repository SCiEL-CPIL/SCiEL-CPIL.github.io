"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+8)-ES with per-dimension adaptive step sizes and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Helper to check time
        def time_ok():
            return time.time() < deadline

        # We'll track the best solution found overall
        best_x = None
        best_f = np.inf

        # Phase 1: Multi-restart exploration with (1+8)-ES
        # We use a fraction of the budget for exploration, rest for refinement
        # But we need to be careful about time. Let's do adaptive allocation.
        
        # Strategy:
        # 1. Start with a global search using multiple restarts of (1+8)-ES
        # 2. Keep the best point found
        # 3. Do a local refinement (CMA-ES-like or fine-grained ES) around the best point
        
        # Let's structure it as:
        # - Allocate ~70% of budget to global exploration with restarts
        # - Allocate ~30% to local refinement
        
        # However, with a time deadline, we should be adaptive.
        # Let's use a simpler approach: run (1+8)-ES with restarts until budget or time runs out,
        # keeping the best point. Then do a local search around the best point with remaining budget/time.
        
        # Actually, let's do a more structured approach:
        # Phase 1: Global search with multiple restarts of (1+8)-ES
        # Phase 2: Local refinement using a smaller step size ES around the best point
        
        # For Phase 1, we'll do several restarts. Each restart is a (1+8)-ES run.
        # We'll allocate budget per restart.
        
        # Let's try: 5 restarts, each with budget/5 evaluations
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if not time_ok():
                break
            
            # Initialize center for this restart
            if r == 0:
                center = rng.uniform(-5, 5, dim)
            else:
                # For subsequent restarts, we can start from a random point or near the best
                # Let's use random points to ensure diversity
                center = rng.uniform(-5, 5, dim)
            
            # Per-dimension step sizes
            sigma = np.full(dim, 1.5)
            
            fc = func(center)
            if fc < best_f:
                best_f = fc
                best_x = center.copy()
            
            for _ in range(budget_per_restart - 1):
                if not time_ok():
                    break
                
                # Generate 8 candidates
                improved = False
                for j in range(8):
                    if not time_ok():
                        break
                    
                    y = center + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    
                    if fy < fc:
                        # Update step sizes based on improvement
                        # Increase sigma where we improved, decrease elsewhere
                        for d in range(dim):
                            if abs(y[d] - center[d]) > 1e-10:
                                if y[d] > center[d]:
                                    sigma[d] *= 1.2
                                else:
                                    sigma[d] *= 1.2
                        # Actually, let's use a simpler adaptation
                        # If we improved, increase sigma slightly
                        sigma *= 1.1
                        # But cap it
                        sigma = np.clip(sigma, 0.01, 5.0)
                        
                        center = y
                        fc = fy
                        improved = True
                        
                        if fc < best_f:
                            best_f = fc
                            best_x = center.copy()
                        break  # Only one improvement per generation (1+8)
                
                if not improved:
                    sigma *= 0.85
                    sigma = np.clip(sigma, 0.01, 5.0)
        
        # Phase 2: Local refinement around the best point found
        if best_x is not None and time_ok():
            # Use remaining budget for local search
            # Estimate remaining budget based on time
            # For simplicity, use a fixed number of local search steps
            # Let's do a fine-grained (1+1)-ES with small steps
            
            x = best_x.copy()
            fx = best_f
            sigma_local = np.full(dim, 0.5)
            
            # Estimate how many steps we can afford
            # Assume ~1ms per function call, so we can do ~10000 calls in 10s
            # But we have limited time. Let's just run until time is up.
            
            local_calls = 0
            max_local_calls = 200  # Conservative estimate
            
            for _ in range(max_local_calls):
                if not time_ok():
                    break
                if local_calls >= budget:
                    break
                
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                local_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_local *= 1.1
                    sigma_local = np.clip(sigma_local, 0.001, 2.0)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.9
                    sigma_local = np.clip(sigma_local, 0.001, 2.0)
        
        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
