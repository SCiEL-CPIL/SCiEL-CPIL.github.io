import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Generate the Halton sequence for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i = i // base
        return result

    def _halton_sequence(self, count, dim, bases):
        """Generate a sequence of 'count' points in 'dim' dimensions."""
        points = np.zeros((count, dim))
        for i in range(1, count + 1):
            for j, base in enumerate(bases):
                points[i - 1, j] = self._halton(i, base)
        return points

    def __call__(self, func):
        # Define bases for 5 dimensions (first 5 primes)
        bases = [2, 3, 5, 7, 11]
        
        # Phase 1: Exploration with Halton sequence
        # Use 150 points for exploration
        n_explore = min(150, self.budget)
        
        # Generate Halton points in [0, 1]^dim
        halton_points = self._halton_sequence(n_explore, self.dim, bases)
        
        # Map to [-5, 5]^dim
        halton_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_explore):
            x = halton_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Exploitation with (1+1)-ES
        # Remaining budget
        remaining_budget = self.budget - n_explore
        
        if remaining_budget > 0:
            x = best_x
            fx = best_f
            sigma = 0.5 # Initial step size
            rng = np.random.default_rng(42)
            
            for _ in range(remaining_budget):
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase step size if we made progress
                    sigma *= 1.05
                else:
                    # Decrease step size if we didn't
                    sigma *= 0.95
                    
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
        return best_x, best_f
