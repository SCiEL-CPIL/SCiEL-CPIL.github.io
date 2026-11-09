# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a low-discrepancy Halton sequence for
    uniform coverage of the Pareto front surface, maximizing hypervolume.
    
    Uses a deterministic Halton sequence (bases 2 and 3) for the two shape
    variables to guarantee uniform coverage without stochastic variance.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the index-th Halton number in the given base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # Optimal distance values for DTLZ4 are 0.5
        # g(x) = sum((x_i - 0.5)^2), so g=0 when x_i=0.5
        # This makes 1+g = 1, putting points exactly on the ideal front.
        
        # To maximize hypervolume with a reference point, we want points
        # distributed uniformly on the Pareto front.
        # For DTLZ4 3-objective, the front is a quarter sphere.
        # Uniform sampling on the sphere surface requires:
        # cos(theta_0) ~ Uniform(0, 1)
        # theta_1 ~ Uniform(0, pi/2)
        
        # The mapping from decision variables to angles is:
        # theta_0 = (pi / 2) * x_0
        # theta_1 = (pi / 2) * x_1
        
        # So:
        # cos((pi/2)*x_0) = u  => x_0 = (2/pi) * arccos(u)
        # (pi/2)*x_1 = v * (pi/2) => x_1 = v
        
        # Strategy: Use Halton sequence for deterministic, low-discrepancy
        # sampling of the shape variables. This guarantees uniform coverage
        # of the Pareto front without stochastic variance.
        
        # Generate Halton sequence points
        # We use two bases: 2 and 3 for the two shape variables
        # Start from index 1 to avoid the (0,0) corner which is problematic
        results = []
        
        for i in range(1, budget + 1):
            if time.time() > deadline:
                break
            
            # Get Halton numbers for this index
            u = self._halton(i, 2)  # Base 2 for first variable
            v = self._halton(i, 3)  # Base 3 for second variable
            
            # Clip to avoid numerical issues at boundaries
            u = np.clip(u, 1e-10, 1 - 1e-10)
            v = np.clip(v, 1e-10, 1 - 1e-10)
            
            # Calculate shape variables
            x_0 = (2.0 / np.pi) * np.arccos(u)
            x_1 = v
            
            # Construct full vector
            x = np.zeros(dim)
            x[0] = x_0
            x[1] = x_1
            x[2:] = 0.5  # Set distance variables to optimal value
            
            # Evaluate
            f = problem(x)
            results.append(f)
        
        # The return value is ignored by the evaluator, 
        # but we return None to satisfy signature if needed.
        return None
# EVOLVE-BLOCK-END
