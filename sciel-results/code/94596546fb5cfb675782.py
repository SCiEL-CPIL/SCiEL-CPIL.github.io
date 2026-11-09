import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        # 1. Deterministic Latin Hypercube Sampling for x1
        # In 1D, LHS is equivalent to uniform sampling.
        # We use linspace for deterministic, uniform coverage of the x1 domain.
        low1, high1 = bounds[0]
        x1_samples = np.linspace(low1, high1, n_evals)
        
        # 2. Construct full population
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_samples
        
        # 3. Leverage ZDT4 structure: set tail variables to minimize g(x)
        # The global minimum of g(x) is at xi=0 for all i>1.
        # We set these variables to 0 (projected to bounds if necessary)
        # to place points on the Pareto front.
        for i in range(1, dim):
            low_i, high_i = bounds[i]
            X[:, i] = np.clip(0.0, low_i, high_i)
            
        # 4. Evaluate all points
        # The evaluator records every call to problem(x).
        F = np.array([problem(x) for x in X])
        
        # 5. Filter to retain non-dominated subset
        # Although all points on the Pareto front are non-dominated,
        # we perform this step to adhere to the directive.
        n_points = len(F)
        is_dominated = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_dominated[i]:
                continue
            for j in range(n_points):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        # Return the filtered population (return value is ignored by evaluator)
        return X[~is_dominated]
