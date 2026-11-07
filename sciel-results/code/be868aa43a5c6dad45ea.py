"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB f15 (Rotated Rastrigin).
    
    Combines:
    1. Multi-Start (1+1)-ES for global basin identification.
    2. Rastrigin-aware Lattice Snapping to exploit integer minima structure.
    3. Fine-grained (1+1)-ES for final continuous refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Multi-Start Exploration ---
        # We use a diverse set of starting points to ensure we find the global basin.
        # f15 has a single global minimum but many local minima.
        
        starts = [np.zeros(dim)]
        
        # Add random starts within the likely region [-4, 4]
        n_random = 8
        for _ in range(n_random):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Add deterministic diverse points to cover corners/midpoints
        starts.append(np.full(dim, 3.0))
        starts.append(np.full(dim, -3.0))
        starts.append(np.array([3.0, -3.0, 3.0, -3.0, 3.0])[:dim])
        starts.append(np.array([-3.0, 3.0, -3.0, 3.0, -3.0])[:dim])

        # Budget allocation for Phase 1
        # We want to spend enough time to find the basin, but save time for refinement.
        # Let's allocate ~60% of budget to exploration.
        exploration_budget = int(budget * 0.6)
        
        # Distribute exploration budget among starts
        # Ensure at least a few calls per start
        calls_per_run = max(10, exploration_budget // len(starts))
        
        for start_point in starts:
            if calls >= exploration_budget or time.time() > deadline:
                break
            
            # (1+1)-ES from this start
            x = start_point.copy()
            fx = eval_func(x)
            
            sigma = 1.5  # Start with moderate step
            window = 10
            recent = [False] * window
            idx = 0
            
            # Run ES for allocated calls or until time/budget runs out
            for _ in range(calls_per_run):
                if calls >= exploration_budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                # Adapt sigma
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.15
                    elif sr < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.05, 5.0)

        # --- Phase 2: Rastrigin-Aware Lattice Refinement ---
        # The Rastrigin function has minima at integer coordinates.
        # Even with rotation, the global minimum is likely close to the integer lattice
        # in the transformed space, or we can search locally around the best point
        # by snapping coordinates to integers.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Generate candidates by snapping each dimension to nearest integers
            # and checking neighbors (integer - 1, integer, integer + 1)
            # We do this iteratively or in a single pass? 
            # Single pass over all combinations is $3^D$ which is $3^5=243$. 
            # This is feasible within the remaining budget (~200 calls).
            
            candidates = set()
            
            # First, snap all dimensions to nearest integer
            base_snap = np.round(x_local)
            candidates.add(tuple(np.clip(base_snap, -5, 5)))
            
            # Then, for each dimension, try integer - 1 and integer + 1
            for d in range(dim):
                for offset in [-1, 0, 1]:
                    cand = base_snap.copy()
                    cand[d] = base_snap[d] + offset
                    cand = np.clip(cand, -5, 5)
                    candidates.add(tuple(cand))
            
            # Also, try snapping only one dimension at a time from the current best_x
            # This helps if the other dimensions are already well-optimized but one is off.
            for d in range(dim):
                for offset in [-1, 0, 1]:
                    cand = x_local.copy()
                    cand[d] = round(cand[d]) + offset
                    cand = np.clip(cand, -5, 5)
                    candidates.add(tuple(np.round(cand, 10))) # Round to avoid float issues in set

            # Evaluate candidates
            # We sort candidates by distance to current best_x to prioritize closer ones?
            # Or just evaluate all. 243 calls is fine.
            for cand in candidates:
                if calls >= budget or time.time() > deadline:
                    break
                y = np.array(cand)
                fy = eval_func(y)
                if fy < f_local:
                    f_local = fy
                    x_local = y.copy()

        # --- Phase 3: Intensive Fine-Grained Refinement ---
        # After snapping to integers, the point is very close to the minimum.
        # Use a small sigma ES to fine-tune.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            sigma = 0.1  # Small step for fine tuning
            window = 5
            recent = [False] * window
            idx = 0
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.1
                    elif sr < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
