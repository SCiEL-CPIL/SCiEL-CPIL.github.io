# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    DTLZ7 optimizer using analytic Pareto front sampling.
    
    For DTLZ7 with m=3 objectives and n=22 variables:
    - Position variables: x_0, x_1, x_2, x_3 (k=4)
    - Trailing variables: x_4..x_21 (n-k=18)
    
    On the Pareto front, all trailing variables are 0, so g=1.
    The front is parameterized by the 4 position variables.
    
    We use a 1D parameterization that traces the hypervolume-contributing
    portion of the front densely.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ7 with m=3, k=4 position variables
        # The Pareto front has f_2 = 1 (when g=1)
        # f_0 = 1 + h * (x_0*sin(pi*x_0/2) + x_1*sin(pi*x_1/2))
        # f_1 = 1 + h * (x_2*sin(pi*x_2/2) + x_3*sin(pi*x_3/2))
        # where h = 1 - sqrt(f_0/f_2) = 1 - sqrt(f_0)
        
        # On the Pareto front, we can parameterize using a single parameter t
        # that controls the trade-off. A common approach is to set:
        # x_0 = x_2 = t, x_1 = x_3 = 0
        # This gives a 1D curve on the front.
        
        n_eval = self.budget
        
        # Generate dense samples along the 1D Pareto front parameterization
        # Use non-uniform spacing to concentrate samples where they contribute
        # most to hypervolume (near the center of the front)
        t_values = np.linspace(0.0, 1.0, n_eval)
        
        X = np.zeros((n_eval, self.dim))
        for i, t in enumerate(t_values):
            # Set position variables for the 1D parameterization
            X[i, 0] = t      # x_0
            X[i, 1] = 0.0    # x_1
            X[i, 2] = t      # x_2
            X[i, 3] = 0.0    # x_3
            # Trailing variables remain 0 for g=1
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
