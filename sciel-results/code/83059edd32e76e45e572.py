import numpy as np
import time

class Optimizer:
    """
    Hybrid Deterministic Local Search with Stochastic Escapes for BBOB f24.
    
    Structural change from (1+1)-ES:
    - Uses deterministic coordinate/pattern search for local exploitation.
    - Uses stochastic jumps only for escaping local minima.
    - Multi-scale step size management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls = 0

        # Strategy:
        # 1. Try to find the global basin quickly using a few broad initializations.
        # 2. Use deterministic local search to converge rapidly within a basin.
        # 3. If local search stalls, jump to a new location.
        
        # Parameters
        h_init = 1.5
        h_min = 1e-5
        h_max = 5.0
        h_shrink = 0.5
        h_grow = 1.5
        stall_threshold = 10 # No improvement in N local search steps
        
        # List of initial points to try. 
        # Lunacek's global minimum is often near the origin in the unshifted frame, 
        # but shifted. We try center and some random points.
        
        def run_local_search(x, h, max_iters, func, calls, deadline, best_f, best_x):
            """
            Perform deterministic local search from x with initial step h.
            Returns (x, f, h, calls, best_f, best_x, improved)
            """
            nonlocal best_f, best_x
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            no_improve = 0
            improved = False
            
            for _ in range(max_iters):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Coordinate descent style: try moving along each axis
                found_improve_in_cycle = False
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    y = x.copy()
                    y[i] += h
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    
                    if fy < f:
                        x = y
                        f = fy
                        found_improve_in_cycle = True
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        # Optionally shrink h slightly or keep it. 
                        # Keeping it might allow larger moves if gradient is consistent.
                        # Let's shrink a bit to be safe, but not too much.
                        h = max(h * 0.9, h_min) 
                    else:
                        # Try negative direction
                        y = x.copy()
                        y[i] -= h
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls += 1
                        
                        if fy < f:
                            x = y
                            f = fy
                            found_improve_in_cycle = True
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            h = max(h * 0.9, h_min)
                
                if found_improve_in_cycle:
                    no_improve = 0
                    # If we improved, maybe try to increase h a bit? 
                    # No, usually we shrink to refine. But if we are far away, we might want to grow.
                    # Let's stick to shrinking for local refinement.
                else:
                    no_improve += 1
                    # Shrink step size
                    h *= h_shrink
                    if h < h_min:
                        break
                    
                    if no_improve >= stall_threshold:
                        break
                        
            return x, f, h, calls, best_f, best_x, improved

        # Phase 1: Initial Exploration
        # We divide the budget into chunks for different initializations.
        # Since we have 500 calls, we can afford ~10-20 restarts.
        # Early restarts should be broad, later ones can be more focused if we find a good region.
        
        # Define a few deterministic starting points + random ones
        start_points = []
        # Center
        start_points.append(np.zeros(n))
        # Small random offsets
        for _ in range(5):
            start_points.append(rng.uniform(-2, 2, n))
        # Larger random offsets
        for _ in range(5):
            start_points.append(rng.uniform(-5, 5, n))

        # Shuffle to mix deterministic and random? 
        # Actually, let's try the center first as it's a common heuristic for BBOB.
        # Then random.
        
        for start_idx, x0 in enumerate(start_points):
            if calls >= budget or time.time() > deadline:
                break
            
            # Allocate remaining budget proportionally? 
            # Or just run until stall/budget.
            # Let's run local search with a generous iteration limit.
            max_iters = 1000 # Will break on stall or budget
            
            x_curr = x0.copy()
            h_curr = h_init
            
            # Run local search
            x_curr, f_curr, h_curr, calls, best_f, best_x, improved = run_local_search(
                x_curr, h_curr, max_iters, func, calls, deadline, best_f, best_x
            )
            
            # If we improved, great. 
            # If we didn't improve at all in this whole local search phase, it was a wasted effort 
            # but we used up some budget.
            
            # If we stalled (h < h_min or no_improve high), we are done with this basin.
            # The next start point will be a new basin.
            
            # Optional: If we are close to the end, do a final polish from the global best
            if calls > budget * 0.8:
                # Do a quick local search from the best found so far with small h
                h_polish = h_min * 10
                run_local_search(best_x, h_polish, 50, func, calls, deadline, best_f, best_x)
                break

        # Final check: ensure we have a valid best_x
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
