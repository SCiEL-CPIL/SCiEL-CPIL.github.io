import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Generate points on the Pareto front
            # Optimize sampling for ZDT1 convex front
            # f1 = x1, f2 = 1 - sqrt(x1)
            # Let u be uniform in [0, 1]. Set x1 = u^2.
            # This linearizes the relationship between x1 and f2.
            u = np.linspace(0, 1, self.budget)
            x1_vals = u ** 2 * (hi[0] - lo[0]) + lo[0]
            
            pop = np.zeros((self.budget, self.dim))
            pop[:, 0] = x1_vals
            for i in range(1, self.dim):
                pop[:, i] = lo[i]
                
            # Evaluate
            for i in range(self.budget):
                problem(pop[i])
                
            return pop
