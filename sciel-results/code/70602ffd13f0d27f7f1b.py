import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 25
        # Number of generations
        G = (self.budget // N) - 1
        
        # Generate weight vectors
        weights = np.random.dirichlet(np.ones(self.n_obj), size=N)
        
        # Initialize population
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        
        # MOEA/D loop
        for g in range(G):
            for i in range(N):
                # Select parent
                j = np.random.choice(N)
                # Crossover
                if np.random.random() < 0.9:
                    child1, child2 = self.sbx(X[i], X[j], lo, hi)
                else:
                    child1, child2 = X[i].copy(), X[j].copy()
                # Mutation
                child1 = self.polynomial_mutation(child1, lo, hi)
                child2 = self.polynomial_mutation(child2, lo, hi)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                
                # Update
                if self.is_better(f1, F, i, weights[i]):
                    X[i] = child1
                    F[i] = f1
                if self.is_better(f2, F, i, weights[i]):
                    X[i] = child2
                    F[i] = f2
        
        return X

    def sbx(self, x1, x2, lo, hi):
        eta = 20
        child1 = x1.copy()
        child2 = x2.copy()
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1 = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                    c2 = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                    
                    child1[i] = np.clip(c1, lo[i], hi[i])
                    child2[i] = np.clip(c2, lo[i], hi[i])
        return child1, child2

    def polynomial_mutation(self, x, lo, hi):
        eta = 20
        for i in range(self.dim):
            if np.random.random() < 1.0 / self.dim:
                y_l = (x[i] - lo[i]) / (hi[i] - lo[i])
                y_u = 1 - y_l
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u + (1 - 2 * u) * y_l ** (eta + 1)) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u) + (2 * u - 1) * y_u ** (eta + 1)) ** (1 / (eta + 1))
                x[i] = np.clip(x[i] + delta * (hi[i] - lo[i]), lo[i], hi[i])
        return x

    def is_better(self, f, F, i, w):
        def tcheb(f, w):
            return np.max(w * (f - np.min(F, axis=0)))
        
        return tcheb(f, w) < tcheb(F[i], w)
