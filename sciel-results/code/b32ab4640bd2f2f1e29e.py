"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simulated Annealing based optimizer with adaptive reheat for AOCC optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 300s to be safe.
        deadline = start_time + 300.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(123)

        def make_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Initialize
        # Start at 0 is a good heuristic for BBOB f18 often, but random is safer against shifts.
        # However, f18 is Schaffers F7 ill-conditioned. It has many local minima.
        # We start at 0.
        current_x = np.zeros(self.dim)
        current_f = make_call(current_x)
        
        if current_f is None:
             # Fallback if budget is 0 or something weird happened
             return best_x, best_f

        # Simulated Annealing Parameters
        # T0: Initial temperature. Needs to be large enough to accept uphill moves initially.
        # The scale of f18 is usually around 100.
        T = 100.0 
        alpha = 0.995 # Cooling factor. Slow cooling allows more exploration.
        min_T = 1e-6
        
        # Step size strategy:
        # Start with a large step to explore the [-5, 5] box quickly.
        # Shrink step size as T drops.
        # In 5D, a step of 2.0 is significant.
        base_step = 2.0
        min_step = 1e-4
        
        # Reheat logic:
        # If we stagnate (no improvement for N steps), we reheat to escape local minima.
        # This is crucial for AOCC because getting stuck in a local minimum early 
        # ruins the "early progress" score.
        
        no_improve_count = 0
        stagnation_limit = 20 # If no improvement in 20 steps, reheat
        
        # Adaptive step size based on acceptance rate? 
        # For simplicity and speed, we link step size to T.
        # step = base_step * (T / T0) ** 0.5 ? Or just linear?
        # Let's try step proportional to T, but with a floor.
        
        # To maximize AOCC, we want to find *a* good solution quickly.
        # SA is good at this if parameters are right.
        
        # We will run until budget or time is exhausted.
        # Since each step is O(1) function call + O(D) numpy ops, it's very fast.
        # 500 calls is tiny. We can easily afford complex logic.
        
        while call_count < self.budget:
            if time.time() > deadline:
                break
                
            # Calculate current step size
            # As T decreases, step size decreases.
            # T starts at 100, ends at 1e-6.
            # We want step to go from ~2.0 to ~1e-4.
            # log scale?
            # Let's use: step = min_step + (base_step - min_step) * (T / T0)
            # This is linear in T.
            step_size = min_step + (base_step - min_step) * (T / 100.0)
            
            # Generate neighbor
            # Gaussian perturbation
            direction = rng.standard_normal(self.dim)
            # Normalize to unit vector to make step_size consistent magnitude?
            # Or just standard normal * step_size?
            # Standard normal * step_size is standard.
            # But in high dims, ||z|| ~ sqrt(D). So effective step is step_size * sqrt(D).
            # For D=5, sqrt(5) ~ 2.23.
            # So effective step is ~2.23 * step_size.
            # If step_size is 2.0, effective is 4.4. That's large.
            # Let's scale direction to unit norm to control step magnitude precisely.
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction[0] = 1.0
            
            new_x = current_x + step_size * direction
            new_x = np.clip(new_x, -5, 5)
            
            new_f = make_call(new_x)
            if new_f is None:
                break
            
            # Acceptance criterion
            if new_f < current_f:
                # Improvement
                current_x = new_x
                current_f = new_f
                no_improve_count = 0
            else:
                # Worse
                delta_f = new_f - current_f
                # Accept with probability exp(-delta_f / T)
                # Avoid log(0) or underflow
                if T > 0:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        current_x = new_x
                        current_f = new_f
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            # Update Temperature
            T *= alpha
            
            # Reheat if stagnated
            if no_improve_count >= stagnation_limit:
                # Reheat to a moderate temperature to escape
                # Don't go all the way back to T0, maybe 10% of T0?
                # Or just 10.0?
                T = 10.0 
                no_improve_count = 0
                # Also, maybe randomize position slightly? No, SA handles that.
                # But if we are stuck in a very deep local min, SA might need a bigger kick.
                # Let's add a "jump" if we reheat multiple times?
                # For now, simple reheat.
                
            # Stop if T is very low and we haven't improved in a while?
            # The loop condition handles budget.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
