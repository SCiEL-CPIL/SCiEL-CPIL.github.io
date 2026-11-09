import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Multi-start (1+1)-ES strategy
        # We will run K independent (1+1)-ES instances.
        # Each instance maintains its own x, sigma, and success history.
        
        # Number of parallel runs
        K = 5
        
        # Initialize K starting points
        # 1. Origin
        # 2-4. Random points in [-4, 4]
        # 5. Point near typical Schwefel minimum (4.20968) with noise
        starts = []
        starts.append(np.zeros(d))
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, d))
        starts.append(4.20968 + rng.normal(0, 1.0, d))
        starts = [np.clip(s, -5, 5) for s in starts]
        
        # State for each (1+1)-ES
        # x: current position
        # sigma: step size
        # success_window: list of last N successes (True/False) to adapt sigma
        # We use a window of size 5 for adaptation, as suggested by insights.
        window_size = 5
        
        es_states = []
        for x_start in starts:
            try:
                f = safe_func(x_start)
            except StopIteration:
                break
            es_states.append({
                'x': x_start.copy(),
                'sigma': 1.0, # Initial step size
                'f': f,
                'success_history': []
            })
            
        if not es_states:
            return best_x, best_f

        # Main loop: Round-robin among active (1+1)-ES instances
        # We stop if budget is exhausted or time limit reached.
        
        # We can also terminate an instance if it stagnates significantly
        # or if its sigma becomes too small.
        
        while calls < budget and time.time() < deadline:
            for state in es_states:
                if calls >= budget or time.time() > deadline:
                    break
                
                x = state['x']
                sigma = state['sigma']
                f_curr = state['f']
                
                # Generate candidate
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                x_candidate = np.clip(x_candidate, -5, 5)
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                
                # Check success
                if f_candidate < f_curr:
                    # Success
                    state['x'] = x_candidate
                    state['f'] = f_candidate
                    state['success_history'].append(True)
                else:
                    # Failure
                    state['success_history'].append(False)
                
                # Adapt sigma based on success rate in the last `window_size` steps
                if len(state['success_history']) >= window_size:
                    recent = state['success_history'][-window_size:]
                    success_rate = np.mean(recent)
                    
                    # Target success rate is typically 1/ln(d+1) ~ 0.4 for (1+1)-ES
                    # But for robustness, we can use a simpler rule:
                    # If success rate > 0.5, increase sigma.
                    # If success rate < 0.5, decrease sigma.
                    # More specifically, aim for ~1/3 or 1/2 success rate.
                    
                    if success_rate > 0.5:
                        # Too many successes, increase step size
                        state['sigma'] *= 1.25
                    elif success_rate < 0.5:
                        # Too many failures, decrease step size
                        state['sigma'] *= 0.8
                    else:
                        # Keep sigma stable
                        pass
                    
                    # Limit sigma to prevent explosion or vanishing
                    state['sigma'] = np.clip(state['sigma'], 1e-6, 10.0)
                    
                    # Optional: Reset history to avoid old data biasing adaptation
                    # state['success_history'] = []
        
        return best_x, best_f
