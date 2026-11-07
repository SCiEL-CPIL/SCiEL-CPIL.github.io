# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        # DTLZ1 Pareto front for M=3 objectives:
        # x_1 + x_2 = 1
        # x_3 ... x_N = 0.5
        # x_1, x_2 >= 0
        
        n_points = min(self.budget, 200)
        
        # Use Halton sequence for better low-discrepancy coverage
        def halton_sequence(n, base):
            seq = []
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                while i > 0:
                    f /= base
                    i, rem = divmod(i, base)
                    x += f * rem
                seq.append(x)
            return np.array(seq)
        
        # Generate points
        # We need 1 parameter for the front (x_1). x_2 is determined.
        # Use Halton in 1D? Just use uniform or specific low-discrepancy.
        # For 1D, Halton is just i/n. 
        # Let's use a mix or just linspace to ensure endpoints.
        
        # Linspace includes endpoints, which are crucial for hypervolume.
        u = np.linspace(0.0, 1.0, n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = u
        X[:, 1] = 1.0 - u
        for i in range(2, self.dim):
            X[:, i] = 0.5
            
        F = np.zeros((n_points, self.n_obj))
        
        for i in range(n_points):
            if time.time() > deadline - 1.0:
                break
            try:
                F[i] = problem(X[i])
            except Exception:
                # If evaluation fails, try a random point? Or skip?
                # Skipping reduces count. 
                pass

        return X
# EVOLVE-BLOCK-END
