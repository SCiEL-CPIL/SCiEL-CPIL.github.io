"""Rewritten optimizer for BBOB-style f13 (Sharp ridge) using multi-start strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start optimizer with parallel (1+1)-ES on top-5 basins."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global multi-start exploration
        # Directive: initialize 50 random points, select top 5
        n_starts = 50
        n_top = 5
        
        if budget < n_starts:
            n_starts = budget // 2
            n_top = max(1, n_starts // 2)
            
        points = []
        values = []
        
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            
        values = np.array(values)
        points = np.array(points)
        
        # Select top n_top points
        top_indices = np.argsort(values)[:n_top]
        top_points = points[top_indices]
        top_values = values[top_indices]
        
        calls_used = n_starts
        remaining_budget = budget - calls_used
        
        if remaining_budget <= 0:
            best_idx = np.argmin(top_values)
            return top_points[best_idx], top_values[best_idx]
            
        # Phase 2: Parallel (1+1)-ES from top 5 starting points
        # Allocate remaining budget evenly
        budget_per_es = remaining_budget // n_top
        
        # State for each ES run
        es_states = []
        for i in range(n_top):
            es_states.append({
                'x': top_points[i].copy(),
                'fx': top_values[i],
                'sigma': 2.0,  # Initial step size
                'stall_count': 0
            })
            
        # Track the best overall
        best_overall_x = es_states[0]['x'].copy()
        best_overall_fx = es_states[0]['fx']
        for s in es_states[1:]:
            if s['fx'] < best_overall_fx:
                best_overall_fx = s['fx']
                best_overall_x = s['x'].copy()
                
        # Run (1+1)-ES with success-based step size adaptation
        # We iterate in rounds until budget is exhausted
        # To keep it simple and evenly allocated, we do one step per ES per "round"
        # But since budget_per_es might not be integer divisible perfectly or we might want to stop early,
        # we just loop while calls_used < budget.
        
        # We'll maintain a success history for adaptation
        # Standard (1+1)-ES adaptation: 
        # c_sigma = (1 + 2 * max(0, sqrt((N+1)/2) - 1)) / (N + 5)
        # N = dim
        N = dim
        c_sigma = (1 + 2 * max(0, np.sqrt((N + 1) / 2) - 1)) / (N + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(N - 1) - 1)
        
        # For simplicity and robustness on small dims, we use a simpler adaptation rule
        # Target success rate ~0.2
        # If success rate > 0.3, increase sigma. If < 0.1, decrease.
        # We'll track a rolling window of the last 15 evaluations for each ES.
        
        success_windows = {i: [] for i in range(n_top)}
        
        while calls_used < budget:
            # Determine which ES to update in this step
            # To ensure even allocation, we can cycle through them.
            # However, simple round-robin is fine.
            # We'll pick the ES with the fewest updates so far in this phase? 
            # Or just cycle 0..4.
            # Let's just cycle.
            
            # Actually, the directive says "allocating the remaining budget evenly".
            # So we should do exactly budget_per_es steps for each.
            # Let's track steps taken per ES.
            
            if not hasattr(self, '_steps_taken'):
                self._steps_taken = [0] * n_top
            else:
                # Reset if previous run left state? No, __init__ is called fresh.
                pass
                
            # Find an ES that hasn't reached its quota
            active_indices = [i for i in range(n_top) if self._steps_taken[i] < budget_per_es]
            
            if not active_indices:
                break
                
            # Pick one randomly from active to avoid bias if order matters? 
            # Or just take the first one to keep it deterministic/simple.
            # Let's take the first one for simplicity, effectively round-robin.
            idx = active_indices[0]
            
            s = es_states[idx]
            
            # Generate candidate
            y = np.clip(s['x'] + s['sigma'] * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1
            self._steps_taken[idx] += 1
            
            if fy < s['fx']:
                s['x'] = y
                s['fx'] = fy
                s['stall_count'] = 0
                success_windows[idx].append(1)
                # Update best overall
                if fy < best_overall_fx:
                    best_overall_fx = fy
                    best_overall_x = y.copy()
            else:
                s['stall_count'] += 1
                success_windows[idx].append(0)
                
            # Keep window size 15
            if len(success_windows[idx]) > 15:
                success_windows[idx] = success_windows[idx][-15:]
                
            # Adapt sigma
            if len(success_windows[idx]) >= 5:
                success_rate = np.mean(success_windows[idx])
                # Target 0.2
                if success_rate > 0.3:
                    s['sigma'] *= 1.1
                elif success_rate < 0.1:
                    s['sigma'] *= 0.9
                # Clamp sigma
                s['sigma'] = np.clip(s['sigma'], 0.01, 10.0)
                
            # Restart if stalled
            if s['stall_count'] > 30:
                # Move towards global best
                # If we are far from global best, jump closer
                dist_to_best = np.linalg.norm(s['x'] - best_overall_x)
                if dist_to_best > 1.0:
                    # Move halfway towards best
                    s['x'] = 0.5 * s['x'] + 0.5 * best_overall_x
                    s['x'] = np.clip(s['x'], -5, 5)
                    # Reset sigma to medium
                    s['sigma'] = 1.0
                    s['stall_count'] = 0
                    success_windows[idx] = []
                    
        return best_overall_x, best_overall_fx
# EVOLVE-BLOCK-END
