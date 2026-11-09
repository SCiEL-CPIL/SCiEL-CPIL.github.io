import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # --- Phase 1: Structured Exploration ---
        # We use a structured set of points to quickly locate the basin of attraction.
        # This replaces the single random start with a small, high-value exploration phase.
        
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Axis-aligned points
        for d in range(dim):
            for dist in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1, -1]:
                    p = np.zeros(dim)
                    p[d] = sign * dist
                    candidates.append(p)
        
        # 2D plane points
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for dist in [1.5, 2.5]:
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            p = np.zeros(dim)
                            p[d1] = s1 * dist
                            p[d2] = s2 * dist
                            candidates.append(p)
        
        # Random points
        for _ in range(10):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        # Shuffle to avoid bias towards origin if it's not optimal
        rng.shuffle(candidates)
        
        # Limit exploration budget to ~20% of total
        explore_budget = min(100, budget // 5)
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        for x_cand in candidates:
            if calls >= explore_budget:
                break
            if time.time() > deadline:
                break
                
            x_cand = np.clip(x_cand, -5, 5)
            fx = func(x_cand)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_cand.copy()
        
        if best_x is None:
            # Fallback if no candidates were evaluated (should not happen)
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1
            
        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Use the adaptive step size strategy from the current program (score 0.7760)
        # which outperformed the fixed/1/5 rule variations in the inspirations.
        
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0  # Start with a moderate step size
        
        # Success/failure tracking for adaptive sigma
        # We use a simple ratio-based adjustment similar to the current program
        # but with slightly tuned constants for the "Attractive Sector" landscape.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                # Increase sigma on success
                sigma *= 1.2
            else:
                # Decrease sigma on failure
                sigma *= 0.9
            
            # Keep sigma in reasonable range
            if sigma > 4.0:
                sigma = 4.0
            elif sigma < 1e-5:
                sigma = 1e-5
        
        # --- Phase 3: Deterministic Coordinate-Ascent Polish ---
        # If we have remaining budget and time, perform a precise coordinate ascent.
        # This leverages the "attractive sector" structure where the minimum is often
        # sharp along specific axes.
        
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline - 5.0:
            # Use a small fraction of remaining budget for polish
            polish_budget = min(remaining_budget, 50)
            
            # Adaptive step size for coordinate ascent
            step = max(sigma * 0.5, 0.01)
            
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                    
                improved = False
                for d in range(dim):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try moving in positive direction
                    x_trial = best_x.copy()
                    x_trial[d] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < best_fx:
                        best_x = x_trial
                        best_fx = f_trial
                        improved = True
                    else:
                        # Try moving in negative direction
                        x_trial = best_x.copy()
                        x_trial[d] -= step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < best_fx:
                            best_x = x_trial
                            best_fx = f_trial
                            improved = True
                    
                    if improved:
                        step *= 0.8  # Shrink step if we found an improvement
                        break  # Restart coordinate loop if we improved
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
        
        return best_x, best_fx
