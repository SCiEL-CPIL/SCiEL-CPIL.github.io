"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic multi-scale local refinement for BBOB f4 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility, but random enough to avoid bad corners
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # 1. Initialization: Sample a few points to find a good starting basin
        # Rastrigin has many local minima. We need to land in a good one.
        # We try a few random points and keep the best.
        init_candidates = []
        # Try center
        init_candidates.append(np.zeros(dim))
        # Try 5 random points
        for _ in range(5):
            init_candidates.append(rng.uniform(-4, 4, dim))
        
        current_x = np.zeros(dim)
        current_f = np.inf
        for p in init_candidates:
            f_p = wrapped_func(p)
            if f_p < current_f:
                current_f = f_p
                current_x = p.copy()
        
        # 2. Multi-scale Refinement
        # The function is Rastrigin: f(x) = sum(x_i^2 - 10*cos(2*pi*x_i)) + 10*n
        # The global minimum is at x=0 (shifted/rotated in the task).
        # Local minima are at integer coordinates.
        # Strategy:
        # Start with a step size large enough to jump between local minima if needed,
        # but small enough to converge to the local minimum center.
        # However, simple coordinate descent can get stuck on the "Rastrigin" ridges.
        # A better approach for Rastrigin is to use a step size that allows
        # "hopping" to adjacent integer centers if the current one is worse,
        # or just converging to the local center.
        
        # Given the rotation/shift, the "integer" grid is rotated.
        # We don't know the grid. But we know the periodicity is roughly 2*pi in the rotated space?
        # Actually, the BBOB f4 is the shifted/rotated Rastrigin.
        # The distance between local minima is 1 unit in the transformed space.
        # In the original space, the distance is also 1 (rotation preserves distance).
        
        # So, a step size of 0.5 is good for local convergence.
        # To escape local minima, we might need larger steps, but Rastrigin is very "spiky".
        # Often, just finding the right local minimum (the global one) is the key.
        # The global minimum is unique.
        
        # Let's use a shrinking step size strategy.
        # Start with a step of 1.0 to allow movement between adjacent minima.
        # Then shrink to 0.1, 0.01, etc.
        
        step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        for step in step_sizes:
            if call_count >= budget or time.time() > deadline:
                break
            
            # For each dimension, try moving +/- step
            # We do multiple passes to ensure convergence
            # Number of passes depends on step size. Smaller steps need more passes?
            # Or just iterate until no improvement?
            # Let's do a fixed number of passes per step size to control budget.
            
            # Estimate remaining budget
            remaining = budget - call_count
            if remaining <= 0:
                break
            
            # For large steps, fewer passes. For small steps, more passes.
            # But we have a hard budget.
            # Let's just loop until no improvement or budget/time runs out.
            
            improved = True
            while improved and call_count < budget and time.time() < deadline:
                improved = False
                for d in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_plus = current_x.copy()
                    x_plus[d] += step
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = wrapped_func(x_plus)
                    if f_plus < current_f:
                        current_f = f_plus
                        current_x = x_plus
                        improved = True
                        continue # Restart dimension loop? Or continue? 
                        # Usually restart is safer for coordinate descent to handle correlations.
                        # But for speed, let's just continue to next dimension.
                        # Actually, if we move in d, other dimensions might become suboptimal.
                        # Let's restart the inner loop (all dimensions) if we improve.
                        # This is expensive. 
                        # Alternative: just continue.
                        # Given the budget, let's be careful.
                        
                    # Try negative step
                    x_minus = current_x.copy()
                    x_minus[d] -= step
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = wrapped_func(x_minus)
                    if f_minus < current_f:
                        current_f = f_minus
                        current_x = x_minus
                        improved = True
        
        return best_x, best_f

# EVOLVE-BLOCK-END
