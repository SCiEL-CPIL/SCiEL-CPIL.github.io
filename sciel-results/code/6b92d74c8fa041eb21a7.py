"""Multi-Start (1+1) Evolution Strategy Optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start (1+1) Evolution Strategy.
    
    Strategy:
    1. Partition the 300 evaluation budget into 30 independent restarts.
    2. Each restart uses exactly 10 evaluations.
    3. Within each restart, perform a (1+1) hill-climb with a fixed, decaying step size.
    4. The step size starts large to cover the domain and decays exponentially to refine.
    5. This parallel-in-time approach maximizes coverage of the search space 
       (especially for multimodal functions like Rastrigin and Gallagher) 
       while maintaining low per-iteration overhead for fast early progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        
        # Configuration for Multi-Start (1+1) ES
        n_restarts = 30
        evals_per_restart = 10
        
        # Ensure we don't exceed budget
        # If budget is not divisible, we might have some extra evals at the end, 
        # but for this strategy, fixed partitioning is simpler and more predictable.
        # We will run exactly n_restarts * evals_per_restart evals if possible.
        
        # Step size schedule:
        # Start with a step that covers a good portion of the domain [-5, 5]
        # Decay it to allow fine-grained search in later restarts or within a restart?
        # The prompt suggests "fixed-step-size hill-climb ... without any global step-size adaptation".
        # However, a single fixed step size for the whole run is suboptimal.
        # The strategy description says: "dedicating exactly 10 evaluations to each for a rapid, fixed-step-size hill-climb".
        # This implies the step size is fixed *for that restart*. 
        # Should the step size change between restarts? 
        # "without any global step-size adaptation" suggests no adaptive mechanism based on success/failure.
        # But a deterministic schedule is not "adaptation". 
        # Let's use a deterministic decay of the base step size across restarts to balance exploration and exploitation.
        # Restart 0: Large step (Exploration)
        # Restart 29: Small step (Exploitation around global best)
        
        base_step_start = 4.0
        base_step_end = 0.1
        
        for r in range(n_restarts):
            if budget <= 0 or time.time() - start_time > time_limit:
                break
            
            # Determine step size for this restart
            # Linear interpolation of log-step size? Or linear step size?
            # Log-linear is often better for optimization scales.
            t = r / max(1, n_restarts - 1)
            log_step = np.log(base_step_start) + t * (np.log(base_step_end) - np.log(base_step_start))
            step_size = np.exp(log_step)
            
            # Choose starting point for this restart
            if r == 0:
                # First restart: start at origin (common heuristic) or random?
                # Origin is deterministic and often close for BBOB.
                # But "hidden seeded shift" means origin might be far.
                # Let's start at origin for the very first one to establish a baseline quickly.
                x = np.zeros(dim)
            else:
                # Subsequent restarts: 
                # Strategy says "independent random restarts".
                # But "inter-restart communication" is forbidden.
                # However, we know best_x_global. 
                # A pure random restart ignores past info. 
                # A "Multi-Start" often implies random starts.
                # Let's stick to the directive: "30 independent random restarts".
                # So, sample uniformly from the box [-5, 5].
                x = np.random.uniform(-5, 5, dim)
            
            # Evaluate starting point
            x = np.clip(x, -5, 5)
            if budget <= 0:
                break
            f_x = func(x)
            budget -= 1
            
            if f_x < best_f_global:
                best_f_global = f_x
                best_x_global = x.copy()
            
            # Perform (1+1) hill-climb for this restart
            # Fixed step size for this restart: step_size
            for k in range(evals_per_restart - 1): # We already used 1 eval, so 9 left
                if budget <= 0 or time.time() - start_time > time_limit:
                    break
                
                # Generate a random direction
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                
                # Propose new point
                x_new = x + step_size * direction
                x_new = np.clip(x_new, -5, 5)
                
                if budget <= 0:
                    break
                f_new = func(x_new)
                budget -= 1
                
                # Accept if better
                if f_new < f_x:
                    x = x_new
                    f_x = f_new
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                # Else: keep x and f_x (standard (1+1) ES behavior)
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
