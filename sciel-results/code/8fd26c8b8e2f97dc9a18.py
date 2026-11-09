"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-restart CMA-ES (simplified 1+1 with covariance) for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def check_time():
            return (time.time() - start_time) < time_limit

        # Global best tracking
        x_global_best = None
        f_global_best = np.inf
        calls = 0

        # Helper to evaluate and update global best
        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Initial random sampling to find a promising region
        # f22 is multimodal. We spend ~10% of budget on random search.
        n_init = max(10, budget // 10)
        
        # First point
        x0 = rng.uniform(-5, 5, dim)
        evaluate(x0)
        
        for _ in range(n_init - 1):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Multi-restart Adaptive (1+1)-ES with Covariance
        # We restart the local optimizer several times, each starting from the current global best
        # but with a fresh, larger step size to escape local minima.
        
        # Parameters for the (1+1)-ES
        # Standard CMA-ES uses a population, but a well-tuned (1+1) with covariance 
        # is very fast and effective for low dimensions like 5.
        # We implement a simplified CMA-ES style adaptation:
        # x_{k+1} = x_k + sigma * z, where z ~ N(0, C)
        # Update C based on successful steps.
        
        max_restarts = 5
        # Budget per restart: decrease as we go? Or fixed?
        # Let's use a fixed fraction of remaining budget for each restart, 
        # but ensure the last restart uses the rest.
        
        for restart in range(max_restarts):
            if not check_time() or calls >= budget:
                break
            
            # Initialize local state
            x = x_global_best.copy()
            f = f_global_best
            
            # Step size: Start somewhat large to allow movement
            # Dim=5. Box size 10.
            sigma = 2.0
            
            # Covariance matrix: Start with identity
            C = np.eye(dim)
            
            # Memory for covariance update
            # Simple update: if step is successful, C = C + alpha * (z z^T - C)
            # If unsuccessful, do nothing or slight decay? 
            # Let's use a simple adaptive scheme:
            # Keep a history of last successful direction.
            
            # To make it more robust, we can use a simplified version of CMA-ES logic
            # without the full population.
            
            # Time/Budget allocation for this restart
            # We want to spend a significant chunk. 
            # Let's say 15% of total budget per restart, min 20 calls.
            restart_budget = max(20, budget // (max_restarts + 1))
            local_calls = 0
            local_limit = min(restart_budget, budget - calls)
            
            # Adaptive sigma
            # If we get stuck (no improvement for X steps), reduce sigma.
            # If we improve a lot, keep or increase.
            
            stall_count = 0
            stall_threshold = 10
            
            while local_calls < local_limit and check_time() and calls < budget:
                # Sample direction
                z = rng.multivariate_normal(np.zeros(dim), C)
                # Normalize z to have unit norm to control step size via sigma
                norm_z = np.linalg.norm(z)
                if norm_z > 1e-10:
                    z = z / norm_z
                
                x_trial = x + sigma * z
                f_trial = evaluate(x_trial)
                local_calls += 1
                
                if f_trial < f:
                    # Success
                    # Update covariance: align with successful direction
                    # C = (1 - 1/dim) * C + (1/dim) * (z z^T)
                    # This is a simple rank-1 update towards the successful direction
                    C = (1 - 1.0/dim) * C + (1.0/dim) * (np.outer(z, z))
                    
                    x = x_trial
                    f = f_trial
                    stall_count = 0
                    
                    # Optional: Increase sigma slightly if we are making big jumps?
                    # Keep sigma constant for now, let C handle the shape.
                else:
                    # Failure
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        # Reduce step size
                        sigma *= 0.7
                        stall_count = 0
                        # Also reset C to identity if we are stuck in a weird direction?
                        # Maybe just decay C towards identity
                        C = 0.5 * C + 0.5 * np.eye(dim)

            # If we still have budget and time, and we haven't used all restarts,
            # the next restart will start from x_global_best again.
            # To encourage exploration, we could perturb the start of the next restart.
            # But x_global_best is the best found. 
            # To escape the local minimum around x_global_best, the next restart
            # should start with a larger sigma.
            # We already reset sigma to 2.0 at the start of the loop.

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
