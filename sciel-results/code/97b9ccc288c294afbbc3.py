import numpy as np
import time

class Optimizer:
    """
    Multi-Start Basin Hopping Optimizer.
    
    This strategy uses a population of 'basins' (local search trajectories).
    Each basin performs a local search with an adaptive step size.
    If a basin stagnates, it is replaced by a new random point (restart).
    The step size for all basins is adapted globally based on the success rate
    of the last generation of steps across all basins.
    
    This is structurally different from (1+1)-ES because:
    1. It maintains multiple independent lineages (basins) simultaneously.
    2. It uses a global success-rate based step-size adaptation (similar to 
       CMA-ES principles but applied to a restart-based local search).
    3. It explicitly models the problem as a collection of basins of attraction,
       restarting basins that appear trapped.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        # Parameters
        N_BASINS = 10       # Number of parallel basins
        MAX_STEPS_PER_BASIN = 20 # Max steps before forced restart if no improvement
        INITIAL_SIGMA = 1.5
        MIN_SIGMA = 1e-3
        MAX_SIGMA = 5.0
        
        rng = np.random.default_rng(12345)
        
        # Initialize basins
        # Each basin is a dict with x, fx, sigma, stagnation_count
        basins = []
        calls_used = 0
        
        # We need to be careful with the budget. 
        # We will evaluate one step per basin per "generation".
        # Total budget is 300.
        
        # Initialize first batch
        for i in range(N_BASINS):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            basins.append({
                'x': x,
                'fx': fx,
                'sigma': INITIAL_SIGMA,
                'stagnation': 0,
                'best_fx': fx
            })
            
        # Global tracking
        global_best_fx = min(b['fx'] for b in basins) if basins else np.inf
        global_best_x = min(basins, key=lambda b: b['fx'])['x'].copy() if basins else None
        
        # Adaptive step size state
        # We track success rate over the last K evaluations
        success_history = []
        K = 50 # Window for success rate
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # Determine how many basins we can update in this "generation"
            # We want to spend roughly 1 call per basin per iteration if possible,
            # but we are limited by budget.
            
            # To ensure we use the budget efficiently, we just pick the next basin to update
            # in a round-robin fashion, or simply iterate through all basins.
            # Given N=10 and Budget=300, we can do ~30 generations of 10 steps.
            
            gen_successes = 0
            gen_total = 0
            
            for i in range(N_BASINS):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                basin = basins[i]
                
                # Generate candidate
                # Use the basin's current sigma
                y = basin['x'] + basin['sigma'] * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_used += 1
                gen_total += 1
                
                # Update basin state
                if fy < basin['fx']:
                    basin['x'] = y
                    basin['fx'] = fy
                    basin['stagnation'] = 0
                    gen_successes += 1
                    
                    if fy < basin['best_fx']:
                        basin['best_fx'] = fy
                    
                    if fy < global_best_fx:
                        global_best_fx = fy
                        global_best_x = y.copy()
                        
                    # Increase sigma slightly on success? 
                    # In basin hopping, we usually keep sigma or increase it 
                    # to explore the basin. But if we are converging, we decrease.
                    # Let's adapt sigma based on global success rate.
                else:
                    basin['stagnation'] += 1
                    # If stagnant for too long, restart this basin
                    if basin['stagnation'] > MAX_STEPS_PER_BASIN:
                        # Restart
                        basin['x'] = rng.uniform(-5, 5, self.dim)
                        basin['fx'] = func(basin['x'])
                        calls_used += 1
                        basin['stagnation'] = 0
                        basin['sigma'] = INITIAL_SIGMA
                        
                        if basin['fx'] < basin['best_fx']:
                            basin['best_fx'] = basin['fx']
                        if basin['fx'] < global_best_fx:
                            global_best_fx = basin['fx']
                            global_best_x = basin['x'].copy()
                        
                        # Note: The restart call is an extra call. 
                        # We should check budget again.
                        if calls_used >= self.budget:
                            break

                # Record success for global adaptation
                success_history.append(1 if fy < basin['fx'] else 0)
                if len(success_history) > K:
                    success_history.pop(0)

            # Adapt global sigma based on success rate
            if gen_total > 0:
                success_rate = np.mean(success_history) if success_history else 0.5
                # Target success rate is ~0.2 for (1+1)-ES, but for population 
                # maybe higher? Let's aim for 0.2-0.3.
                # If success rate is high, increase sigma.
                # If success rate is low, decrease sigma.
                
                # Apply adaptation to all basins
                for basin in basins:
                    if success_rate > 0.3:
                        basin['sigma'] *= 1.1
                    elif success_rate < 0.1:
                        basin['sigma'] *= 0.9
                    else:
                        # Slight decay to converge
                        basin['sigma'] *= 0.98
                    
                    # Clip sigma
                    basin['sigma'] = np.clip(basin['sigma'], MIN_SIGMA, MAX_SIGMA)

        return global_best_x, global_best_fx
