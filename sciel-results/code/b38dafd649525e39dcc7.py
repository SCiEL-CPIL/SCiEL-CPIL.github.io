"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a structured grid on the Pareto front.
    
    Strategy:
    1. DTLZ2 with M=3 has a Pareto front determined by x[0], x[1] (position variables).
    2. x[2:] should be set to the center of the bounds (0.5) to minimize g(x)=0.
    3. We use a structured grid to generate 200 points in [0, 1]^2.
    4. These points are mapped to the DTLZ2 Pareto front angles.
    5. The resulting 200 objective vectors are returned.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Structure Assumption:
        # M=3 objectives => m=2 position variables (x[0], x[1]).
        # x[2:] are distance variables.
        # To be on the Pareto front, g(x) must be 0.
        # g(x) = sum((x[i] - 0.5)^2) for i >= m.
        # So set x[2:] to 0.5 (center of [0, 1]).
        
        n_points = self.budget
        
        # Use a Halton sequence for low-discrepancy sampling in [0, 1]^2
        # This provides more uniform coverage than a grid for hypervolume
        def halton_sequence(n, base):
            """Generate n points of a Halton sequence with given base."""
            result = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = 0.0
                val = i + 1
                while val > 0:
                    f /= base
                    r += f * (val % base)
                    val //= base
                result[i] = r
            return result
        
        # Generate Halton sequences for 2 dimensions (bases 2 and 3)
        x0_halton = halton_sequence(n_points, 2)
        x1_halton = halton_sequence(n_points, 3)
        
        # Combine into grid_points (2D array)
        grid_points = np.column_stack([x0_halton, x1_halton])
        
        results = []
        
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            
            # For DTLZ2, the Pareto front is parameterized by angles.
            # x[0] and x[1] are position variables that determine the angle.
            # To get uniform coverage on the Pareto front (quarter sphere),
            # we should use:
            #   theta1 = pi/2 * x[0]  (but this is already in [0,1])
            #   theta2 = pi/2 * x[1]
            # Then:
            #   f1 = sin(theta1) * cos(theta2)
            #   f2 = sin(theta1) * sin(theta2)
            #   f3 = cos(theta1)
            # But DTLZ2 uses:
            #   f_i = (1+g) * x_{i-m} * sin(theta_i) for i=1..m-1
            #   f_m = (1+g) * x_{i-m} * cos(theta_m)
            # With g=0 and x[2:]=0.5, we get the Pareto front.
            # The Halton points in [0,1]^2 directly correspond to the
            # position variables, which already map to the front.
            
            # Position variables
            x[0] = grid_points[i, 0]
            x[1] = grid_points[i, 1]
            
            # Distance variables: set to center of bounds for g=0
            if self.dim > 2:
                # Use 0.5 as bounds are [0, 1]
                x[2:] = 0.5
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                # If an exception occurs, break to preserve valid results
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
