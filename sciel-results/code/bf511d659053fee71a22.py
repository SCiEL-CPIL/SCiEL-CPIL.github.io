"""Multi-Start Simulated Annealing with Adaptive Cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin well under 300s/330s
        
        d = self.dim
        budget = self.budget
        
        if budget <= 0:
            return np.zeros(d), np.inf

        global_best_x = np.zeros(d)
        global_best_f = np.inf
        calls_used = 0
        
        # Use a fixed seed for reproducibility of the random number generator 
        # within the run, but the function instances are shifted, so 
        # the relative quality of starts might vary. 
        # However, standard practice in these competitions often allows 
        # using the default RNG or a fixed one. The prompt says "hidden seeded shift", 
        # implying the function is fixed per instance. 
        # We will use np.random.default_rng() for better quality random numbers.
        rng = np.random.default_rng()

        # Strategy: Multi-Start Simulated Annealing
        # 1. Generate a set of diverse initial points.
        # 2. For each point, run a SA loop with a specific budget allocation.
        # 3. SA uses a heavy-tailed proposal (Cauchy) to escape plateaus.
        # 4. Cooling schedule is adaptive: if no improvement for a while, cool faster; 
        #    if improving, cool slower (or restart temperature).
        
        # Number of independent SA runs
        num_starts = 8
        
        # Allocate budget: 
        # Reserve some calls for the initial evaluations.
        # Each SA run gets roughly (budget - num_starts) / num_starts calls.
        # But we want to be flexible. Let's just run until budget is exhausted 
        # or time is up, cycling through starts if one finishes early (unlikely with SA).
        
        # Initial points: Uniform in [-4.5, 4.5] to stay safely inside [-5, 5]
        # and cover the likely region of the minimum.
        initial_points = []
        for i in range(num_starts):
            if calls_used >= budget:
                break
            x0 = rng.uniform(-4.5, 4.5, d)
            f0 = func(x0)
            calls_used += 1
            initial_points.append((x0, f0))
            
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()

        if not initial_points:
            return global_best_x, global_best_f

        # Sort initial points by fitness to prioritize better starts
        initial_points.sort(key=lambda x: x[1])

        # SA Parameters
        # Initial temperature should be high enough to accept uphill moves.
        # The step function has flat regions, so the "energy" difference is 0 
        # until the drop. The drop magnitude is unknown but likely O(1) to O(100).
        # Let's start T = 10.0 and adapt.
        
        # Proposal distribution: 
        # Standard Gaussian might get stuck in local plateaus if sigma is too small.
        # If sigma is too large, it's random search.
        # Cauchy distribution has heavy tails, good for exploration.
        # We will use a mix: mostly Gaussian with a "jump" probability for Cauchy.
        
        # Let's implement a single SA trajectory function
        def run_sa_trajectory(x_start, f_start, max_calls):
            nonlocal global_best_f, global_best_x, calls_used
            
            x = x_start.copy()
            f = f_start
            
            # Local best for this trajectory
            local_best_x = x.copy()
            local_best_f = f
            
            # Temperature
            T = 5.0
            min_T = 1e-4
            
            # Step size for Gaussian component
            sigma_g = 1.0
            # Scale for Cauchy component
            sigma_c = 2.0
            
            # Stall counter
            stall_count = 0
            target_improvements = 5 # If we improve 5 times, reset stall
            
            while calls_used < budget and calls_used < max_calls:
                if time.time() > deadline:
                    break
                
                # Generate proposal
                # 10% chance of a big jump (Cauchy), 90% local move (Gaussian)
                if rng.random() < 0.1:
                    # Cauchy noise: rng.standard_cauchy is available in newer numpy
                    # To be safe, use 1/rng.standard_normal() which is Cauchy(0,1)
                    noise = rng.standard_cauchy(d) / sigma_c
                else:
                    noise = rng.standard_normal(d) * sigma_g
                
                x_new = x + noise
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                calls_used += 1
                
                # Acceptance criterion
                delta_f = f_new - f
                if delta_f < 0:
                    # Improvement
                    x = x_new
                    f = f_new
                    stall_count = 0
                    # Adapt step size: slightly expand if we found a new best? 
                    # Or shrink? In SA, we usually keep step size constant or adapt 
                    # based on acceptance rate. 
                    # For step functions, if we just jumped onto a plateau, 
                    # we want to explore that plateau.
                    # Let's shrink sigma_g slightly to explore the new local region.
                    sigma_g *= 0.95 
                    sigma_c *= 0.95
                    
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                            # Reset temperature on global improvement to explore more
                            T = min(T * 2.0, 10.0) 
                            stall_count = 0
                else:
                    # Worse or equal
                    # Accept with probability exp(-delta_f / T)
                    if T > 0 and delta_f < 1000: # Avoid overflow
                        p_accept = np.exp(-delta_f / T)
                    else:
                        p_accept = 0.0
                    
                    if rng.random() < p_accept:
                        x = x_new
                        f = f_new
                        # Accepting uphill moves is how we escape.
                        # Keep step size similar
                        sigma_g *= 1.01
                        sigma_c *= 1.01
                    else:
                        # Rejected
                        pass
                
                # Cooling
                # Adaptive cooling:
                # If we haven't improved in a while, cool down to focus.
                # If we are improving, keep temperature higher to explore.
                if stall_count > 10:
                    T *= 0.9
                elif stall_count > 20:
                    T *= 0.8
                else:
                    T *= 0.98 # Slow cooling
                
                T = max(T, min_T)
                
                # If temperature is very low and no improvement for a long time, 
                # it's effectively local search. We might break to save budget for other starts.
                if T < 1e-3 and stall_count > 50:
                    break

            return local_best_x, local_best_f

        # Run SA trajectories
        # We have allocated 'max_calls' implicitly by the global budget check.
        # To ensure diversity, we can cap the calls per trajectory if we have many starts,
        # but since SA is stochastic and might run long, we just let it run until 
        # budget is exhausted or it decides to stop.
        # However, if one trajectory takes all 500 calls, we don't get the benefit of multi-start.
        # Let's cap each trajectory to use at most 1/3 of the remaining budget if we have >2 starts left.
        
        remaining_starts = len(initial_points)
        for i, (x0, f0) in enumerate(initial_points):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Estimate calls left
            calls_left = budget - calls_used
            # If we have many starts left, limit this run
            if remaining_starts > 1:
                # Allow this run to use at most half the remaining budget, 
                # ensuring at least some calls for future starts
                max_calls_for_this = min(calls_left, int(calls_left / remaining_starts) * 2 + 10)
            else:
                max_calls_for_this = calls_left
            
            run_sa_trajectory(x0, f0, max_calls_for_this)
            remaining_starts -= 1

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
