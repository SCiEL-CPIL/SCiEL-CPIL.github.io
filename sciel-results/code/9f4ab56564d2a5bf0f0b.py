"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start basin identification + local refinement for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # Safety margin before 300s limit

        # Phase 1: Systematic multi-start to identify best basin (60% of budget)
        # Rastrigin: f(x) = 10*dim + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global min at x*=0 (shifted by hidden offset). Period is 2 in each dim.
        # Sample a grid of candidate centers to find which basin is lowest.
        
        # Generate candidate centers: combine structured grid + random samples
        # Coarse grid: sample every 2.0 units (one period) plus offsets
        n_coarse = 5  # points per dim in coarse grid -> 5^5 = 3125, too many
        # Use a smarter approach: Halton-like quasi-random + structured offsets
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # Mix of: (a) structured lattice at period boundaries, (b) random
        n_starts = min(200, int(budget * 0.4))  # ~200 starts for 500 budget
        
        starts = []
        # Structured: sample at multiples of 2 (period) with small offsets
        # This ensures we hit near each local minimum
        for offset in [0.0, 1.0]:
            for d in range(dim):
                base = np.zeros(dim)
                # Sample a few points per dimension at period boundaries
                pass
        
        # Simpler: use stratified sampling
        # Divide [-5,5] into 5 intervals of width 2 (one period each)
        # Sample 1-2 points per cell
        n_per_cell = 1
        cells = []
        for i in range(5 ** dim):
            # Convert index to 5-dim coordinates
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % 5)
                tmp //= 5
            # Map to [-5, 5] with stratified sampling
            point = []
            for d in range(dim):
                # Uniform within cell [ -5 + 2*coords[d], -5 + 2*(coords[d]+1) ]
                lo = -5.0 + 2.0 * coords[d]
                hi = lo + 2.0
                point.append(rng.uniform(lo, hi))
            cells.append(np.array(point))
        
        # Take first n_starts from cells (shuffled)
        rng.shuffle(cells)
        starts = cells[:n_starts]
        
        # Add some pure random points for coverage
        n_random = min(50, budget - n_starts - 100)
        for _ in range(n_random):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all starts, keep best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        for s in starts:
            if time.time() > deadline:
                break
            s = np.clip(s, -5, 5)
            f_s = func(s)
            call_count += 1
            if f_s < best_f:
                best_f = f_s
                best_x = s.copy()
        
        # Phase 2: Local refinement from best point (remaining budget)
        # Use adaptive (1+1)-ES with aggressive step size
        remaining = budget - call_count
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate step
            
            for i in range(remaining - 1):
                if i % 5 == 0 and time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                # If we've converged very tightly, try smaller steps
                if sigma < 1e-4:
                    sigma = 1e-4
        
        # Phase 3: If budget remains, try a few more random restarts
        calls_made = call_count + (remaining - 1 if remaining > 10 else 0)
        if calls_made < budget - 5 and time.time() < deadline:
            for _ in range(min(20, budget - calls_made)):
                if time.time() > deadline:
                    break
                s = rng.uniform(-5, 5, dim)
                f_s = func(s)
                if f_s < fx:
                    fx = f_s
                    x = s.copy()
        
        return x, fx
# EVOLVE-BLOCK-END
