"""
Hybrid Optimizer for BBOB f3 (Rastrigin):
1. Structured Exploration: Targets integer coordinates where Rastrigin's global 
   minimum is known to reside (post-shift/rotation).
2. Adaptive Local Refinement: Multi-restart (1+1)-ES with success-rate based 
   step size adaptation to converge to the precise minimum.
3. Budget Management: Explicit deadline guards and call counting to ensure 
   validity and maximize score within the 300s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s, we stop at 280s to be safe.
        deadline = time.time() + 280.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        # Fixed seed for reproducibility of the search strategy itself
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            # Ensure x is within bounds before passing to func (func clips, but good practice)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # ---------------------------------------------------------
        # Phase 1: Structured Exploration
        # Rastrigin's global minimum is at integer coordinates.
        # We sample integer points, near-integer points, and random points.
        # ---------------------------------------------------------
        
        n_total_explore = min(100, budget // 5)
        # Split: 50% integer, 30% near-integer, 20% random
        n_int = int(n_total_explore * 0.5)
        n_near_int = int(n_total_explore * 0.3)
        n_rand = n_total_explore - n_int - n_near_int
        
        candidates = []
        
        # 1. Integer points in [-4, 4]^5 (Minimizer lies in [-4, 4]^5 per contract)
        for _ in range(n_int):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            candidates.append(x_int)
            
        # 2. Near-integer points (Integer + small noise)
        for _ in range(n_near_int):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            offset = rng.uniform(-0.5, 0.5, size=dim)
            x_near = x_int + offset
            candidates.append(np.clip(x_near, -5.0, 5.0))
            
        # 3. Random points for diversity
        for _ in range(n_rand):
            x_rand = rng.uniform(-5.0, 5.0, size=dim)
            candidates.append(x_rand)

        # Evaluate all candidates
        candidate_values = []
        for x_cand in candidates:
            if check_deadline():
                break
            fx = evaluate(x_cand)
            candidate_values.append(fx)
        
        # Sort and select top K for refinement
        if candidate_values:
            sorted_indices = np.argsort(candidate_values)
            # Keep top 5 diverse candidates
            k_top = min(5, len(sorted_indices))
            top_candidates = [candidates[i] for i in sorted_indices[:k_top]]
        else:
            top_candidates = [rng.uniform(-5.0, 5.0, size=dim)]

        # ---------------------------------------------------------
        # Phase 2: Multi-Restart Adaptive (1+1)-ES
        # Refine the top candidates found in Phase 1.
        # ---------------------------------------------------------
        
        remaining_budget = budget - calls_made
        # Budget allocation per restart
        # We want to spend roughly equal time on each top candidate, 
        # but prioritize if one is clearly better? 
        # Simple approach: Fixed steps per restart, limited by total budget.
        
        # Estimate steps per restart:
        # If we have 5 candidates and 400 calls left, we can do ~80 steps each.
        steps_per_restart = max(10, remaining_budget // max(1, len(top_candidates)))
        
        for start_idx, seed in enumerate(top_candidates):
            if check_deadline():
                break
                
            x = seed.copy()
            # Re-evaluate seed to ensure it's current in our local context
            # (It was evaluated in Phase 1, so we can skip if we trust the cache, 
            # but func might be stochastic? No, deterministic. 
            # However, evaluate() updates global best. 
            # We already have the value from Phase 1, but we don't store it per candidate easily.
            # Just re-evaluate to be safe and simple, cost is 1 call.
            fx = evaluate(x)
            
            # Adaptive (1+1)-ES
            sigma = 0.5
            success_count = 0
            total_steps = 0
            no_improve = 0
            max_no_improve = 20  # Early termination if stuck
            
            while not check_deadline() and total_steps < steps_per_restart:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Adapt sigma every 5 steps
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 0.95
                    success_count = 0
                    sigma = np.clip(sigma, 1e-6, 2.0)
                
                # Early termination if stuck
                if no_improve > max_no_improve:
                    break
            
            # Update global best if this local run found something better
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---------------------------------------------------------
        # Phase 3: Final High-Precision Polish
        # If we still have budget, refine the global best further.
        # ---------------------------------------------------------
        
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Start small for high precision
            success_count = 0
            total_steps = 0
            no_improve = 0
            max_no_improve = 50
            
            # Run until deadline or stuck
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Adapt sigma every 5 steps
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.1:
                        sigma *= 0.75
                    else:
                        sigma *= 0.95
                    success_count = 0
                    sigma = np.clip(sigma, 1e-8, 0.5)
                
                # Early termination if stuck
                if no_improve > max_no_improve:
                    break
            
            # Final update
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
