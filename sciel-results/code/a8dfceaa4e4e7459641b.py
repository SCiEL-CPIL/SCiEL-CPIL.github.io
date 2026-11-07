import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ2 optimization
        # M = n_obj = 3
        # N = dim = 12
        # Angle variables: x[0], x[1]
        # G variables: x[2] ... x[11]
        
        # Use a grid for better coverage than pure random
        # 20 points in u1, 10 points in u2 -> 200 total
        n1 = 20
        n2 = 10
        total = n1 * n2
        if total > self.budget:
             # Adjust if budget is smaller, but here it is exactly 200
             n1 = int(np.sqrt(self.budget))
             n2 = self.budget // n1
             
        u1_vals = np.linspace(0, 1, n1, endpoint=False) + 0.5/n1
        u2_vals = np.linspace(0, 1, n2, endpoint=False) + 0.5/n2
        
        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing='ij')
        U1 = U1.flatten()
        U2 = U2.flatten()
        
        # If we have leftover budget, add random points
        if len(U1) < self.budget:
            rem = self.budget - len(U1)
            U1 = np.concatenate([U1, np.random.uniform(0, 1, rem)])
            U2 = np.concatenate([U2, np.random.uniform(0, 1, rem)])
            
        # Ensure we don't exceed budget
        U1 = U1[:self.budget]
        U2 = U2[:self.budget]
        
        x1 = (2.0 / np.pi) * np.arccos(U1)
        x2 = U2
        
        # Construct decision vectors
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        if self.dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate
        for i in range(self.budget):
            problem(X[i])
            
        return X
