"""Rewritten program: coordinate descent with dense 1D golden-section search for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate descent with golden-section 1D line searches.
    
    Exploits the anisotropy of the Bent Cigar function by decoupling
    the narrow valley dimension from the broad ones. Each coordinate is
    optimized independently using a dense golden-section search over
    the box [-5, 5].
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: use a few random starts and pick the best
        n_starts = min(10, budget // 50)
        x = np.empty(dim)
        fx = np.inf
        for _ in range(n_starts):
            x_candidate = rng.uniform(-5, 5, dim)
            fx_candidate = func(x_candidate)
            if fx_candidate < fx:
                fx = fx_candidate
                x = x_candidate.copy()

        # Golden section search on a single coordinate
        # We search over the full range [-5, 5] for each coordinate
        # but we can narrow it down based on current best
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        inv_phi2 = 1.0 - inv_phi  # ~0.382

        def golden_search_1d(x, fx, axis, n_iter):
            """Perform golden-section search along one axis."""
            # Start with the full box
            a = -5.0
            b = 5.0
            
            # Compute initial points
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            x_c = x.copy()
            x_d = x.copy()
            x_c[axis] = c
            x_d[axis] = d
            
            fc = func(x_c)
            fd = func(x_d)
            
            for _ in range(n_iter):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - inv_phi * (b - a)
                    x_c = x.copy()
                    x_c[axis] = c
                    fc = func(x_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_phi * (b - a)
                    x_d = x.copy()
                    x_d[axis] = d
                    fd = func(x_d)
            
            # Best point is the midpoint of the final interval
            best_val = (a + b) / 2.0
            x_best = x.copy()
            x_best[axis] = best_val
            f_best = func(x_best)
            
            return x_best, f_best

        # Main loop: iterate over coordinates, doing golden-section searches
        # We need to be careful about budget usage
        calls_used = n_starts + 1  # initial evaluations + first best
        
        # Strategy: multiple passes over coordinates with decreasing precision
        # Pass 1: coarse search (fewer iterations per coordinate)
        # Pass 2: fine search (more iterations per coordinate)
        
        # Estimate how many function calls per golden search
        # Each golden search iteration uses 1 function call (after initial 2)
        # So a search with n_iter iterations uses n_iter + 2 calls total
        
        # Let's plan: 
        # - 3 passes over dim coordinates
        # - Pass 1: 20 iterations each -> 22 calls per coordinate -> 110 calls
        # - Pass 2: 30 iterations each -> 32 calls per coordinate -> 160 calls  
        # - Pass 3: 40 iterations each -> 42 calls per coordinate -> 210 calls
        # Total: ~480 + 10 starts = 490, within budget
        
        # But let's be more adaptive: use remaining budget
        passes = [20, 30, 40]
        
        for n_iter in passes:
            for axis in range(dim):
                # Check if we have enough budget
                remaining = budget - calls_used
                needed = n_iter + 2
                if remaining < needed:
                    break
                
                x, fx = golden_search_1d(x, fx, axis, n_iter)
                calls_used += n_iter + 2
        
        return x, fx
# EVOLVE-BLOCK-END
