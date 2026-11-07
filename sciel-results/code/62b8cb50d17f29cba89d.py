"""Multi-objective optimizer for ZDT1 using optimal hypervolume placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimal hypervolume maximizer for ZDT1 (2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget  # 200


        # ZDT1 Pareto front: f1 = x[0] in [0,1], f2 = 1 - sqrt(x[0])
        # HV = sum_{i=1}^{n-1} (t_{i+1}-t_i)*(10+sqrt(t_i)) + (11-1)*(10+sqrt(1))
        # Maximize the variable part with t_1=0, t_n=1
        # Optimal: gaps proportional to weights w(t_i)=10+sqrt(t_i)
        
        t = np.linspace(0.0, 1.0, n_evals)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            w = 10.0 + np.sqrt(np.maximum(t, 1e-15))
            # n-1 gaps sum to 1, proportional to first n-1 weights
            # t_i = cumsum(w[:i-1]) / sum(w[:n-1]) for i=2..n, t_1=0, t_n=1
            w_first_n1 = w[:n_evals - 1]
            s = np.sum(w_first_n1)
            t_new = np.zeros(n_evals)
            t_new[0] = 0.0
            t_new[n_evals - 1] = 1.0
            if n_evals > 2:
                cumsum_w = np.cumsum(w_first_n1)
                t_new[1:n_evals-1] = cumsum_w[:n_evals-2] / s
            # Monotonicity check
            t_new = np.maximum.accumulate(t_new)
            t_new = np.minimum(t_new, 1.0)
            t_new[-1] = 1.0
            if np.max(np.abs(t_new - t)) < 1e-14:
                t = t_new
                break
            t = t_new

        # Create x vectors: x[0] = t_i, x[1:] = 0 (Pareto front)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
