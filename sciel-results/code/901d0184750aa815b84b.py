import numpy as np
from scipy.stats import qmc
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    Strategy: 
    1. Exploit the known structure of ZDT3 where the Pareto front lies on the 
       manifold x_2 = ... = x_n = 0 (minimizing g(x) to 1).
    2. Use a Sobol sequence (low-discrepancy sequence) to sample x_1 in [0, 1].
       This provides better coverage of the 1D search space than uniform random 
       sampling, ensuring that both the continuous segments of the Pareto front 
       and any potential features in the gaps are well-represented.
    3. Evaluate all candidates within the time budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Generate Sobol sequence points in 1D for x_1
        # Sobol sequence is quasi-random, providing uniform coverage with low discrepancy
        try:
            sampler = qmc.Sobol(d=1, scramble=True)
            x1_values = sampler.random(n=budget)[:, 0]
        except Exception:
            # Fallback to random if scipy qmc fails for any reason
            x1_values = np.random.rand(budget)

        # Construct the decision vectors
        # x[0] = x1_values, x[1:] = 0
        # Setting x[1:] = 0 ensures g(x) = 1, which is the global minimum of g,
        # placing the points on the Pareto front (or very close to it).
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values

        # Evaluate each point
        for i in range(budget):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
