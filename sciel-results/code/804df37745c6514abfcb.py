"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic multi-scale CMA-ES variant with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Generate a comprehensive set of directions
        directions = []
        # Coordinate axes
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
        # Diagonal directions (all pairs)
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0
                d[j] = 1.0
                directions.append(d)
        # All-one vector
        directions.append(np.ones(dim))
        
        # Normalize directions
        directions = np.array([d / np.linalg.norm(d) for d in directions])
        n_dirs = len(directions)

        # Multi-scale search with adaptive step sizing
        sigma = 4.0
        min_sigma = 1e-5
        scale_decay = 0.5
        plateau_threshold = 1e-10
        restart_count = 0
        max_restarts = 3

        while calls_made < budget and sigma > min_sigma:
            improvements = 0
            best_fx_before = best_fx
            
            # Try each direction at current scale
            for d in directions:
                if calls_made >= budget:
                    break
                    
                # Try positive step
                y = np.clip(x + sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    improvements += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

                # Try negative step
                if calls_made >= budget:
                    break
                y = np.clip(x - sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    improvements += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

            # Adaptive step sizing
            if improvements > 0:
                # Had improvements, keep or slightly increase step
                sigma = min(sigma * 1.1, 4.0)
            else:
                # No improvements, shrink step significantly
                sigma *= scale_decay
                
                # Check for plateau and consider restart
                if best_fx - best_fx_before < plateau_threshold and restart_count < max_restarts:
                    # Try a random restart from a different starting point
                    restart_count += 1
                    rng = np.random.default_rng(restart_count * 12345)
                    x_new = rng.uniform(-4, 4, size=dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    
                    if fx_new < best_fx:
                        x, fx = x_new, fx_new
                        best_x, best_fx = x.copy(), fx
                    # Reset sigma for the new location
                    sigma = 2.0

        # Final fine-grained coordinate search with remaining budget
        remaining = budget - calls_made
        if remaining > 0:
            fine_sigma = 1e-4
            for _ in range(remaining):
                if calls_made >= budget:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] + fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if improved:
                    continue
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] - fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if not improved:
                    fine_sigma *= 0.5
                    if fine_sigma < 1e-7:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
