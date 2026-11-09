import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Hybrid Optimizer: Deterministic Lattice Search followed by (1+1)-ES local refinement.
    
    Phase 1: Deterministic 3^5 lattice search to guarantee global coverage.
    Phase 2: (1+1)-ES with adaptive step size to refine the solution.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        
        # Phase 1: Deterministic lattice search
        # Use a 3^5 grid (243 points) covering [-5, 5]^5
        # This guarantees coverage of the multimodal landscape
        
        # Generate grid points
        grid_values = np.linspace(lower, upper, 3)
        grid_points = np.array(np.meshgrid(*([grid_values] * dim))).T.reshape(-1, dim)
        
        best_x = None
        best_fx = float('inf')
        calls_used = 0
        
        for point in grid_points:
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            fx = func(point)
            calls_used += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = point.copy()
        
        # If we ran out of budget during lattice search, return best found
        if calls_used >= budget or time.time() > deadline:
            return best_x, best_fx
        
        # Phase 2: (1+1)-ES local refinement from best grid point
        x = best_x.copy()
        fx = best_fx
        
        sigma = 1.0
        sigma_min = 0.001
        sigma_max = 5.0
        
        # Adaptive step size parameters
        success_count = 0
        total_attempts = 0
        
        while calls_used < budget and time.time() < deadline:
            # Generate offspring
            z = np.random.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, lower, upper)
            
            fy = func(y)
            calls_used += 1
            total_attempts += 1
            
            # Update
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            
            # Adaptive sigma adjustment
            if total_attempts % 5 == 0:
                success_rate = success_count / total_attempts
                if success_rate > 0.3:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                success_count = 0
                total_attempts = 0
                
            # Clamp sigma
            sigma = np.clip(sigma, sigma_min, sigma_max)
        
        return best_x, best_fx
