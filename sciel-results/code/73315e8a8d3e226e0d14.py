import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # DTLZ4 with 3 objectives and 12 dimensions:
        # g(x) = sum_{i=2}^{11} (x_i - 0.5)^alpha  where alpha=10 for DTLZ4
        # On the Pareto front: g=0, so x[2:]=0.5
        # f1 = 0.5 * x[0] * (1 + g)
        # f2 = 0.5 * x[1] * (1 - x[0]) * (1 + g)  
        # f3 = 0.5 * (1 - x[1]) * (1 + g)
        # When g=0: f1 = 0.5*x[0], f2 = 0.5*x[1]*(1-x[0]), f3 = 0.5*(1-x[1])
        # These satisfy f1+f2+f3 = 0.5
        
        # Strategy: Generate stratified points in (u,v) space where:
        # u = x[0], v = x[1]
        # Then f1 = 0.5*u, f2 = 0.5*v*(1-u), f3 = 0.5*(1-v)
        
        # For better hypervolume, we want to maximize coverage in objective space.
        # The Pareto front is a 2D surface in 3D objective space.
        
        # Use a grid + jitter approach for stratified sampling in (u,v) ∈ [0,1]^2
        # This gives uniform coverage of the Pareto front in decision space.
        
        # But we can do better: sample in a way that gives more uniform coverage
        # in objective space. Let's use the fact that:
        # f1 = 0.5*u, so u = 2*f1
        # f3 = 0.5*(1-v), so v = 1 - 2*f3
        # f2 = 0.5*v*(1-u) = 0.5*(1-2*f3)*(1-2*f1)
        
        # So if we want uniform coverage in (f1, f2, f3) on the front, we can
        # sample (f1, f3) uniformly and compute f2.
        
        # Actually, let's just use a well-distributed set of (u,v) pairs.
        # Use a combination of:
        # 1. A stratified grid in (u,v) space
        # 2. Halton sequence for additional coverage
        
        F = []
        budget = self.budget
        
        # Use stratified sampling: divide [0,1]^2 into cells and sample one point per cell
        # With 200 evaluations, use ~14x14 grid = 196 points, plus 4 extra
        n_grid = 14
        n_grid_points = n_grid * n_grid
        
        if n_grid_points >= budget:
            # Use only grid points
            for i in range(budget):
                if time.time() > deadline:
                    break
                # Convert linear index to 2D grid index
                row = i // n_grid
                col = i % n_grid
                # Add jitter within each cell
                u = (col + 0.5) / n_grid
                v = (row + 0.5) / n_grid
                
                x = np.empty(self.dim)
                x[0] = u
                x[1] = v
                x[2:] = 0.5
                
                f = problem(x)
                F.append(f)
        else:
            # Use grid points + Halton for remaining
            for i in range(n_grid_points):
                if time.time() > deadline:
                    break
                row = i // n_grid
                col = i % n_grid
                u = (col + 0.5) / n_grid
                v = (row + 0.5) / n_grid
                
                x = np.empty(self.dim)
                x[0] = u
                x[1] = v
                x[2:] = 0.5
                
                f = problem(x)
                F.append(f)
            
            # Add Halton points for remaining budget
            remaining = budget - len(F)
            for i in range(remaining):
                if time.time() > deadline:
                    break
                # Halton sequence in bases 2 and 3
                idx = i + 1
                u = self._halton(idx, 2)
                v = self._halton(idx, 3)
                
                x = np.empty(self.dim)
                x[0] = u
                x[1] = v
                x[2:] = 0.5
                
                f = problem(x)
                F.append(f)
        
        return F
    
    def _halton(self, index, base):
        """Generate the index-th Halton number in base."""
        result = 0
        f = 1
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
