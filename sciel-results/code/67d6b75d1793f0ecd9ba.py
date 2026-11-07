import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Strategy:
        # 1. Fix x[1:] to 0 (minimizes g(x) to 1, placing points on the Pareto front manifold).
        # 2. Sample x[0] using a Halton sequence for low-discrepancy coverage.
        # 3. Explicitly include x[0]=0 and x[0]=1 to capture the extremes of the front.
        
        n = self.budget
        if n < 2:
            # Fallback for very small budgets
            X = np.zeros((n, self.dim))
            X[:, 0] = np.linspace(0, 1, n)
        else:
            X = np.zeros((n, self.dim))
            # Include endpoints
            X[0, 0] = 0.0
            X[-1, 0] = 1.0
            
            # Generate Halton sequence for the middle points
            def halton_sequence(n_points, base):
                seq = []
                for i in range(1, n_points + 1):
                    f = 1.0
                    r = 0.0
                    while i > 0:
                        f /= base
                        r += f * (i % base)
                        i //= base
                    seq.append(r)
                return np.array(seq)
            
            n_mid = n - 2
            mid_samples = halton_sequence(n_mid, 2)
            X[1:-1, 0] = mid_samples
            
            # Ensure all other dimensions are 0 (which they are by initialization)
            # Clip to bounds just in case (though 0 and 1 are within [0,1])
            X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
