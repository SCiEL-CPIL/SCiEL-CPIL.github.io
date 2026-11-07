"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with Local Refinement for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Simulated Annealing Parameters
        # f19 is Griewank-Rosenbrock. Range [-5, 5].
        # Initial temperature should be high enough to accept large jumps.
        # Typical function value range for f19 is [0, ~100].
        T = 50.0
        T_min = 0.1
        # Cooling rate: T = T * cooling_rate
        # We want to reach T_min in about 70% of the budget.
        # Let's say SA phase is 350 calls (70% of 500).
        # (T_min / T) = (cooling_rate)^350
        # cooling_rate = (T_min / T)^(1/350)
        sa_steps = int(budget * 0.7)
        cooling_rate = (T_min / T) ** (1.0 / max(1, sa_steps))

        # Adaptive step size
        step_size = 1.0

        # Phase 1: Simulated Annealing
        for i in range(sa_steps):
            if calls >= budget:
                break

            # Generate neighbor
            # Use Gaussian noise scaled by step_size
            noise = rng.standard_normal(n) * step_size
            x_new = x + noise
            x_new = np.clip(x_new, -5, 5)

            fx_new = func(x_new)
            calls += 1

            # Acceptance criterion
            if fx_new < fx:
                x = x_new
                fx = fx_new
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                # Reduce step size slightly on improvement? Or keep it?
                # Usually, we reduce step size as we cool.
            else:
                # Accept worse solution with probability exp(-delta / T)
                delta = fx_new - fx
                if delta < 0: # Should not happen if fx_new > fx
                    delta = 0
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    x = x_new
                    fx = fx_new
                else:
                    # Reject, maybe increase step size if stuck?
                    # For now, just keep current.
                    pass

            # Update temperature
            T *= cooling_rate

            # Adaptive step size: 
            # If we are accepting many moves, maybe step is too small?
            # If we are rejecting many, step might be too large?
            # Simple heuristic: decrease step size as we cool.
            # step_size = max(0.1, step_size * 0.99) 
            # But let's link it to T.
            step_size = max(0.01, 1.0 * (T / T_min)) ** 0.5
            # Actually, let's just scale step size with sqrt(T) or similar.
            # Let's try a fixed decay for step size as well.
            step_size = max(0.05, step_size * 0.98)

        # Phase 2: Local Refinement (Hill Climbing)
        # Use remaining budget for local search around best_x
        # We will try small perturbations in random directions and coordinate axes.
        local_budget = budget - calls
        if local_budget > 0:
            x_local = best_x.copy()
            fx_local = best_f
            # Small initial step for local search
            local_step = 0.1
            
            for i in range(local_budget):
                if calls >= budget:
                    break
                
                # Try a few random directions or coordinate axes
                # To be efficient, let's try one random direction per call
                # Or cycle through axes?
                # Random direction is better for non-orthogonal landscapes.
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                
                x_try = x_local + local_step * direction
                x_try = np.clip(x_try, -5, 5)
                
                fx_try = func(x_try)
                calls += 1
                
                if fx_try < fx_local:
                    x_local = x_try
                    fx_local = fx_try
                    if fx_try < best_f:
                        best_f = fx_try
                        best_x = x_local.copy()
                    # Reduce step size on success
                    local_step *= 0.9
                else:
                    # Try opposite direction? Or just reduce step?
                    # If we fail, reduce step size to find a smaller improvement
                    local_step *= 0.8
                    if local_step < 1e-4:
                        # Restart local search from best_x with new random direction?
                        # Or just break?
                        # Let's try a larger random jump to escape local min?
                        # No, we are in local refinement.
                        # Let's reset step size occasionally
                        if i % 50 == 0:
                            local_step = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
