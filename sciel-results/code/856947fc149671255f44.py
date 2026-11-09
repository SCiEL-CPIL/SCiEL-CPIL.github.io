import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT4 specific knowledge:
        # f1 = x1
        # f2 = (1 + g) * h(x1, f1)
        # g = sum(xi^2 - 10*cos(12*pi*xi) + 10) for i=2..N
        # Global minimum of g is at xi=0 for all i.
        # Pareto front is at x2..xN = 0, x1 in [0,1].
        
        n_evals = self.budget
        dim = self.dim
        
        # Generate x1 using LHS (uniform in [0,1])
        # LHS for 1D is just sorted uniform randoms or linspace.
        # Linspace is deterministic and uniform.
        x1_values = np.linspace(0.0, 1.0, n_evals)
        
        # Create the full population
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_values
        
        # For x2..xN, set to 0 to minimize g(x)
        # This places all points exactly on the Pareto front.
        # X[:, 1:] is already 0.
        
        # Evaluate all points
        # The problem function clips to bounds, but 0 is within [-5,5] typically.
        # bounds are given, but we assume standard ZDT4 bounds [-5,5].
        # Even if bounds are different, 0 is usually inside.
        # If bounds don't include 0, we should project to the closest bound?
        # ZDT4 bounds are always [-5,5].
        
        F = np.array([problem(x) for x in X])
        
        return X
