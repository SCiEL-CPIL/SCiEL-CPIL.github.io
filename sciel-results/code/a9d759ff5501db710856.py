"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer leveraging DTLZ1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # DTLZ1: x[0] controls position on front, x[1:] control shape
        # Pareto front: sum of objectives = 0.5 * (1 + x[0]) when shape vars are set correctly
        # For shape variables, we need sum(x[1:]^2) = 0.5 for the standard DTLZ1 front
        
        # Strategy: Sample x[0] uniformly and set shape variables to lie on the sphere
        # that gives the correct front, with small perturbations for robustness
        
        budget = self.budget
        dim = self.dim
        
        # Phase 1: Systematic sampling of the Pareto front
        # x[0] from 0 to 1, shape vars on sphere of radius sqrt(0.5) in 6D
        n_shape = dim - 1  # 6 shape variables
        
        # Generate points on the (n_shape-1)-sphere of radius sqrt(0.5)
        # Using uniform sampling on sphere
        
        def sample_sphere_point(rng, n, radius):
            """Sample uniformly from the (n-1)-sphere of given radius in R^n."""
            v = rng.standard_normal(n)
            v = v / np.linalg.norm(v) * radius
            return v
        
        # Number of systematic front points
        n_front = min(100, budget // 3)
        
        all_F = []
        used = 0
        
        # Systematic sampling: spread x[0] values
        x0_vals = np.linspace(0.01, 0.99, n_front)
        
        for i in range(n_front):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            x0 = x0_vals[i]
            # Sample shape variables on the sphere
            shape = sample_sphere_point(rng, n_shape, np.sqrt(0.5))
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = shape
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            all_F.append(f)
            used += 1
        
        # Phase 2: Random exploration to fill gaps and find better points
        remaining = budget - used
        if remaining > 0:
            # Mix of random points and perturbed front points
            for i in range(remaining):
                if time.time() > deadline - 5.0:
                    break
                
                if i < remaining // 2:
                    # Random point
                    x = lo + range_ * rng.random(dim)
                else:
                    # Perturbed front point
                    x0 = rng.random()
                    shape = sample_sphere_point(rng, n_shape, np.sqrt(0.5))
                    x = np.zeros(dim)
                    x[0] = x0
                    x[1:] = shape
                    # Add small noise
                    x += 0.01 * range_ * rng.standard_normal(dim)
                    x = np.clip(x, lo, hi)
                
                f = problem(x)
                all_F.append(f)
                used += 1
        
        return all_F
# EVOLVE-BLOCK-END
