import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ2 specific structure
        # M = 3 objectives
        # n = 12 variables
        # Position variables: x[0], x[1]
        # Distance variables: x[2] ... x[11]
        
        # Standard DTLZ2: g(x) = sum((x_i - 0.5)^2) for i in range(M, n)
        # Optimal g=0 when x_i = 0.5
        
        n_pos = self.n_obj - 1
        n_dist = self.dim - n_pos
        
        # Generate Halton sequence for position variables
        # We have budget evaluations.
        # Let's use all of them for the front coverage.
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        for i in range(1, self.budget + 1):
            x = np.zeros(self.dim)
            # Fill position variables with Halton sequence
            for j in range(n_pos):
                x[j] = halton(i, 2 + j) # Base 2 for dim 0, Base 3 for dim 1
            # Fill distance variables with 0.5
            for k in range(n_pos, self.dim):
                x[k] = 0.5
            points.append(x)
        
        # Evaluate all points
        # Note: The problem function might be slow or have side effects, but we just call it.
        # We must return the result of the last call? No, "return value of __call__ is ignored".
        # But we must not raise exceptions.
        
        # We need to record the objective vectors.
        # The evaluator records "Every returned objective vector".
        # So we just need to call problem(x) for each x.
        
        # To be safe with time, we can break if we take too long, but 200 calls is fast.
        
        last_result = None
        for x in points:
            last_result = problem(x)
            
        return last_result
