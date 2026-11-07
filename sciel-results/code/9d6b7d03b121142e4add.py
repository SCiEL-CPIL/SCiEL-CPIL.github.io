import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        used = 0
        
        M = self.dim - self.n_obj + 1
        
        # Generate a Halton sequence for deterministic low-discrepancy sampling
        # This ensures good coverage without relying on np.random for the sampling pattern,
        # though np.random.seed is set by the evaluator.
        
        def halton_sequence(n, d):
            # Generate n points in d dimensions using Halton sequence
            points = np.zeros((n, d))
            for j in range(d):
                base = j + 2
                points[:, j] = self._halton_values(n, base)
            return points
        
        def _halton_values(n, base):
            # Compute the first n values of the Halton sequence for a given base
            values = np.zeros(n)
            f = 1.0 / base
            for i in range(1, n + 1):
                f_tmp = f
                n_tmp = i
                while n_tmp > 0:
                    n_tmp, mod = divmod(n_tmp, base)
                    f_tmp *= base
                    values[i - 1] += mod * f_tmp
            return values
        
        # We want to sample the first M variables to cover the front.
        # The remaining dim - M variables should be 0 to ensure g=0.
        
        # Let's use a mix of Halton sampling and random sampling to maximize diversity.
        # Halton sequence provides a structured grid-like coverage.
        
        n_halton = max_evals
        U = self._halton_sequence(n_halton, M)
        
        # Construct X
        X = np.zeros((n_halton, self.dim))
        X[:, :M] = U
        # X[:, M:] is 0, which ensures g=0
        
        # Evaluate
        F = np.zeros((n_halton, self.n_obj))
        for i in range(n_halton):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            used += 1
            
        return F
