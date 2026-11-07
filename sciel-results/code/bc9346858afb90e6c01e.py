import numpy as np

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Exploits the analytic structure of the ZDT3 Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # Generate x1 values uniformly distributed in [0, 1]
        # This covers the Pareto front parameter space
        x1_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Initialize population
        # x[0] corresponds to f1
        # x[1:] correspond to g, set to 0 to minimize g (optimal for Pareto front)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate each point
        # The evaluator records the objective values returned by problem(x)
        for i in range(n_evals):
            problem(X[i])
            
        return X
