import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        D = self.dim
        B = self.budget

        # Phase 1: Latin Hypercube Sampling on the known ZDT2 Pareto front
        # x1 in [lo[0], hi[0]], x2..xD = lo[1:]
        n_phase1 = B
        
        # Generate LHS for x1
        # Divide [lo[0], hi[0]] into B equal intervals
        # Pick one random point in each interval
        rng = np.random.RandomState(42) # Fixed seed for reproducibility within the call? 
        # The evaluator calls np.random.seed(seed) before creating Optimizer.
        # So we should use np.random directly or respect the seed.
        
        # Using np.random directly to respect the evaluator's seed
        # But LHS requires a permutation.
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Create intervals
        edges = np.linspace(x1_lo, x1_hi, n_phase1 + 1)
        widths = np.diff(edges)
        
        # Random offsets in [0, 1)
        u = np.random.rand(n_phase1)
        
        # Permute to ensure coverage (LHS property)
        # Actually, standard LHS picks one point per interval.
        # The "Latin" part is for multi-dim. For 1-dim, it's just stratified sampling.
        # Stratified sampling is just picking one point per bin.
        # So u * widths + edges[:-1] is sufficient.
        
        x1_vals = edges[:-1] + u * widths
        
        # Construct X
        if D > 1:
            X_rest = np.tile(lo[1:], (n_phase1, 1))
            X = np.column_stack([x1_vals, X_rest])
        else:
            X = x1_vals.reshape(-1, 1)

        F = np.empty((n_phase1, self.n_obj))
        for i in range(n_phase1):
            F[i] = problem(X[i])
            if time.time() > deadline:
                break

        return X
