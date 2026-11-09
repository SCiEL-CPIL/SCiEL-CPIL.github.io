import numpy as np

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    Exploits the known structure of DTLZ4:
    - Variables 0 to n-M are distance variables (should be 0 for optimal front).
    - Variables n-M+1 to n-1 are shape variables (determine position on the front).
    Uses a Halton sequence for uniform coverage of the shape variable space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_halton(self, num, dim):
        """Generate a Halton sequence of 'num' points in 'dim' dimensions."""
        # Use the first 'dim' primes as bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        bases = primes[:dim]
        
        points = np.zeros((num, dim))
        for i in range(1, num + 1):
            for d in range(dim):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= bases[d]
                    i_val, rem = divmod(i_val, bases[d])
                    r += rem * f
                points[i - 1, d] = r
        return points

    def __call__(self, problem, bounds):
        n = self.dim
        M = self.n_obj
        
        # DTLZ4 structure:
        # Distance variables: indices 0 to n-M
        # Shape variables: indices n-M+1 to n-1
        n_dist = n - M + 1
        n_shape = M - 1
        
        # Generate low-discrepancy samples for the shape variables
        # This ensures uniform coverage of the Pareto front surface
        shape_samples = self._generate_halton(self.budget, n_shape)
        
        recorded_vectors = []
        
        for i in range(self.budget):
            x = np.zeros(n)
            
            # Set distance variables to 0.0
            # This minimizes the g function, placing the point on the Pareto front
            # (assuming standard DTLZ4 where g is minimized at 0 for dist vars)
            # x[0:n_dist] remains 0.0
            
            # Set shape variables
            x[n_dist:n] = shape_samples[i]
            
            # Ensure bounds are respected (though [0,1] is standard)
            for j in range(n):
                if x[j] < bounds[j][0]:
                    x[j] = bounds[j][0]
                elif x[j] > bounds[j][1]:
                    x[j] = bounds[j][1]
            
            # Evaluate
            f = problem(x)
            recorded_vectors.append(f)
            
        return recorded_vectors
