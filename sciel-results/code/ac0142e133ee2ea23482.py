import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. 
        # We aim to finish well before. 
        # 500 calls should be very fast in Python/NumPy if func is simple.
        # If func is slow, we need to be careful.
        # Let's set a hard deadline of 320s to be safe, but check time frequently.
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Number of parallel (1+1)-ES lines
        n_agents = 5
        
        # Initialize agents using Latin Hypercube-like sampling
        # For 5 agents in 5D, we can use a simple permutation strategy or just random with stratification.
        # Let's use random but ensure they are spread out.
        rng = np.random.default_rng(42)
        
        agents = []
        # First agent at origin? Or random?
        # Origin is a good bet for many BBOB functions, but f14 might be shifted.
        # Let's use LHS.
        # Generate 5 points in [0, 1]^5 such that each dimension is stratified.
        # For 5 points, we can just permute the indices 0..4 for each dimension.
        # Point i has coordinate j in dimension i? No, that's a diagonal.
        # Let's just use random points in [-4, 4]^5 for simplicity, 
        # but add the origin as one of them to cover the center.
        
        # Agent 0: Origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        record(x0, f0)
        agents.append({
            'x': x0.copy(),
            'f': f0,
            'sigma': 1.0,
            'last_improve_call': calls,
            'stagnation': 0
        })
        
        # Agents 1-4: Random points
        for i in range(1, n_agents):
            if not time_ok() or calls >= budget:
                break
            xi = rng.uniform(-4.0, 4.0, dim)
            fi = func(xi)
            calls += 1
            record(xi, fi)
            agents.append({
                'x': xi.copy(),
                'f': fi,
                'sigma': 1.0,
                'last_improve_call': calls,
                'stagnation': 0
            })
            
        if not agents:
            return best_x, best_f

        # (1+1)-ES Parameters
        # Expansion factor on success
        sigma_expand = 1.05
        # Shrink factor on failure
        sigma_shrink = 0.95
        # Minimum sigma
        sigma_min = 1e-6
        # Maximum sigma
        sigma_max = 10.0
        
        # Stagnation threshold: if no global improvement in 50 calls, restart agent from global best
        stagnation_limit = 50
        
        while calls < budget and time_ok():
            # Select the best agent (Best-First)
            best_agent_idx = 0
            best_agent_val = np.inf
            for i, a in enumerate(agents):
                if a['f'] < best_agent_val:
                    best_agent_val = a['f']
                    best_agent_idx = i
            
            agent = agents[best_agent_idx]
            x = agent['x']
            f = agent['f']
            sigma = agent['sigma']
            
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = func(x_trial)
            calls += 1
            
            if f_trial < f:
                # Accept
                agent['x'] = x_trial
                agent['f'] = f_trial
                # Adapt sigma: expand
                agent['sigma'] = min(sigma * sigma_expand, sigma_max)
                # Reset stagnation for this agent
                agent['stagnation'] = 0
                # Check if global best improved
                if f_trial < best_f:
                    record(x_trial, f_trial)
                    # Reset stagnation counter for all agents? 
                    # Or just track global best improvement time.
                    # For simplicity, we track stagnation per agent relative to its own best.
                    # But for restart, we want to restart if an agent is stuck 
                    # and the global best hasn't improved recently.
                    pass
            else:
                # Reject
                # Adapt sigma: shrink
                agent['sigma'] = max(sigma * sigma_shrink, sigma_min)
                agent['stagnation'] += 1
            
            # Check for restart conditions
            # If an agent is very close to convergence (sigma small) and stagnant,
            # or if it has been stagnant for a long time, restart it.
            if agent['sigma'] < sigma_min * 10 or agent['stagnation'] > stagnation_limit:
                # Restart from global best with small noise
                if best_x is not None:
                    noise = rng.standard_normal(dim) * 0.1
                    x_new = best_x + noise
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < best_f:
                        record(x_new, f_new)
                    agent['x'] = x_new
                    agent['f'] = f_new
                    agent['sigma'] = 0.5 # Reset to a medium step size
                    agent['stagnation'] = 0
        
        return best_x, best_f
