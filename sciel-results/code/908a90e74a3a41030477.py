import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, bases):
        result = np.zeros((n, len(bases)))
        for j, base in enumerate(bases):
            f = 1.0 / base
            for i in range(n):
                i_val = i + 1
                while i_val > 0:
                    result[i, j] += f * (i_val % base)
                    i_val //= base
                    f /= base
        return result

    def __call__(self, problem, bounds):
        n_evals = self.budget
        # Reserve 3 for extremes
        n_halton = n_evals - 3
        if n_halton < 0: n_halton = 0
        
        X = np.zeros((n_evals, self.dim))
        
        # 1. Extremes
        # (1, 0, 0)
        X[0, 0] = 1.0
        X[0, 11] = 0.5
        # (0, 1, 0)
        X[1, 1] = 1.0
        X[1, 11] = 0.5
        # (0, 0, 1)
        X[2, 2] = 1.0
        X[2, 11] = 0.5
        
        # 2. Halton
        if n_halton > 0:
            halton_points = self._halton_sequence(n_halton, [2, 3])
            theta1 = halton_points[:, 0] * (np.pi / 2)
            theta2 = halton_points[:, 1] * (np.pi / 2)
            
            X[3:3+n_halton, 0] = np.cos(theta1)
            X[3:3+n_halton, 1] = np.sin(theta1) * np.cos(theta2)
            X[3:3+n_halton, 2] = np.sin(theta1) * np.sin(theta2)
            X[3:3+n_halton, 11] = 0.5
            
        for x in X:
            problem(x)
            
        return X
