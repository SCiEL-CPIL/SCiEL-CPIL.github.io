"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start Nelder-Mead Optimizer.
    
    Strategy:
    1. Global Phase: Use a deterministic Sobol sequence to generate a set of 
       well-distributed starting points across the domain [-5, 5]^dim.
    2. Local Phase: Run Nelder-Mead simplex optimization from the best starting 
       point found in the global phase.
    3. Restart Phase: If budget remains and the primary Nelder-Mead run stagnates 
       or completes, restart from the second-best starting point, and so on.
    
    This approach is structurally different from stochastic ES because it uses 
    deterministic global sampling for coverage and geometric local search for 
    convergence, avoiding the random walk behavior of ES in multimodal landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim):
        """
        Generate a Sobol sequence of points in [0, 1]^dim.
        Uses a simple bit-reversal method for the first few dimensions.
        For higher dimensions or larger n, this is an approximation.
        """
        # For small dim and n, we can use a simple halving strategy or 
        # rely on numpy's random if SciPy is not strictly required for Sobol, 
        # but the prompt says SciPy is available. 
        # However, to be safe and self-contained, we can use a simple 
        # deterministic grid or a basic Sobol-like sequence.
        # Let's use a simple low-discrepancy sequence approximation.
        
        # Actually, since we have SciPy available, we should use it if possible, 
        # but to avoid import overhead or issues, let's implement a simple 
        # deterministic set of points.
        
        # Simple approach: Use the first n points of a Halton sequence or 
        # just a well-spaced grid.
        # Given budget is 500, we can afford maybe 5-10 starting points.
        
        # Let's generate points using a simple bit-reversal Sobol for dim <= 32.
        # This is a known simple implementation.
        
        def bit_reverse(n, dim):
            # Reverse the bits of n (dim bits)
            res = 0
            for i in range(dim):
                res = (res << 1) | (n & 1)
                n >>= 1
            return res
        
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                # Simple Sobol-like: use bit reversal of (i+1) for direction d
                # This is not true Sobol but provides low discrepancy for small n
                # A better simple method: use the van der Corput sequence for each dim
                # with different bases.
                
                # Van der Corput sequence for dimension d+1 (base 2, 3, 5, ...)
                base = 2 ** (d + 1) # This grows too fast.
                
                # Let's stick to a simpler robust method: 
                # Uniformly distributed points with a jitter, or a grid.
                # Given the small budget, a grid of 2^dim points is too many if dim=5 (32 points).
                # 32 points * ~10 calls each = 320 calls. This is feasible.
                
                # Let's use a simple Latin Hypercube or just random with a fixed seed.
                # To be deterministic and well-distributed, let's use a grid for dim=5.
                # But dim is variable.
                
                # Let's use a simple Halton sequence.
                pass
                
        # Let's implement Halton sequence properly, it's simple and effective.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Bases: first few primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(n):
            for d in range(dim):
                base = primes[d]
                points[i, d] = halton(i + 1, base)
        
        return points

    def _nelder_mead(self, func, x0, budget, deadline):
        """
        Runs Nelder-Mead simplex optimization starting from x0.
        Returns (best_x, best_fx, calls_made).
        """
        dim = self.dim
        # Initialize simplex
        # x0 is the centroid
        simplex = [x0.copy() for _ in range(dim + 1)]
        
        # Perturb vertices
        # Standard initialization: one vertex is x0, others are x0 + step * e_i
        step = 0.05 # Small initial step
        for i in range(dim):
            simplex[i][i] += step
        
        # Evaluate initial simplex
        fvals = [func(x) for x in simplex]
        calls = dim + 1
        
        best_x = x0.copy()
        best_fx = min(fvals)
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = simplex[0].copy()
            
            # Check for convergence (optional, but helps with stagnation)
            # If the range of function values is very small, we might be stuck
            # But for now, let's just run until budget or time is out.
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[0]:
                # 2. Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection is better than worst, but not better than best
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # 3. Contraction
                # Outside contraction if f_r < fvals[-1], inside otherwise
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (centroid - simplex[-1])
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # 4. Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
        
        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Generate starting points using Halton sequence
        # We want to use a portion of the budget for global exploration.
        # Let's say we pick 5 starting points.
        n_starts = 5
        starts = self._sobol_sequence(n_starts, dim)
        
        # Map from [0, 1] to [-5, 5]
        starts = starts * 10 - 5
        
        # Evaluate all starting points to rank them
        f_starts = [func(s) for s in starts]
        calls = n_starts
        
        # Sort starting points by function value
        order = np.argsort(f_starts)
        starts = [starts[i] for i in order]
        f_starts = [f_starts[i] for i in order]
        
        global_best_x = starts[0].copy()
        global_best_fx = f_starts[0]
        
        # Run Nelder-Mead from the best starting point
        # Allocate most of the budget to the first run
        # If it finishes or stagnates, move to the next
        
        # Budget allocation:
        # Run 1: 60% of remaining budget
        # Run 2: 20% of remaining budget
        # Run 3: 10% of remaining budget
        # Run 4: 5% of remaining budget
        # Run 5: 5% of remaining budget
        
        remaining_budget = budget - calls
        allocations = [0.6, 0.2, 0.1, 0.05, 0.05]
        
        for i in range(n_starts):
            if time.time() > deadline or remaining_budget <= 0:
                break
            
            alloc = int(remaining_budget * allocations[i])
            if alloc < 10: # Minimum calls for NM to be useful
                alloc = min(remaining_budget, 10)
                
            # Run Nelder-Mead
            x_opt, f_opt, calls_made = self._nelder_mead(
                func, starts[i], alloc, deadline
            )
            
            remaining_budget -= calls_made
            calls += calls_made
            
            if f_opt < global_best_fx:
                global_best_fx = f_opt
                global_best_x = x_opt.copy()
            
            # Optional: If the best value didn't improve significantly, 
            # we might want to break early if we are stuck.
            # But let's try the next start anyway.
        
        return global_best_x, global_best_fx

# EVOLVE-BLOCK-END
