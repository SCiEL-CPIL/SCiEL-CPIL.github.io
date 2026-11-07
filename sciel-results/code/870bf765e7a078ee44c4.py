"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: radial sweep + aggressive (1+1)-ES for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Radial sweep to identify the anisotropic axis
        # The Discus function has extreme anisotropy: one narrow "needle" axis
        # and wide transverse directions. We sweep radially in multiple directions.
        rng = np.random.default_rng(12345)
        
        # Generate diverse directions: coordinate axes + random
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            d2 = d.copy()
            d2[i] = -1.0
            directions.append(d2)
        
        # Add random directions
        for _ in range(dim * 3):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
                directions.append(d)
        
        # Radial sweep: for each direction, try multiple radii
        radii = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 4.8, 4.95]
        
        for d in directions:
            if calls >= budget:
                break
            for r in radii:
                if calls >= budget:
                    break
                x = r * d
                f = evaluate(x)
        
        # Phase 2: Aggressive (1+1)-ES centered on best point found
        if calls >= budget:
            return best_x, best_f
        
        # Start from best point found in Phase 1
        x_current = best_x.copy()
        f_current = best_f
        
        # Adaptive step size
        sigma = 1.0
        # Very aggressive: high mutation rate, fast adaptation
        c_sigma = 0.5  # Fast sigma adaptation
        c_c = 0.5      # Fast path adaptation
        
        # Path variables for cumulative step adaptation
        p_sigma = np.zeros(dim)
        
        # (1+1)-ES with aggressive parameters
        # High mutation rate: we'll use a large initial sigma and adapt quickly
        while calls < budget:
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x_current + sigma * z
            
            f_trial = evaluate(x_trial)
            
            if f_trial < f_current:
                # Success: increase sigma slightly (exploit more)
                x_current = x_trial
                f_current = f_trial
                # Adaptive sigma: increase on success
                sigma *= 1.05
                sigma = min(sigma, 5.0)
                # Update path
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (x_trial - x_current) / sigma
            else:
                # Failure: decrease sigma (explore less, but keep exploring)
                sigma *= 0.9
                sigma = max(sigma, 0.01)
                # Update path with negative contribution
                p_sigma = (1 - c_sigma) * p_sigma - c_sigma * (x_trial - x_current) / sigma
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.005, 5.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
