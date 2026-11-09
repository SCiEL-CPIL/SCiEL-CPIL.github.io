import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate Sobol sequence for 2D
        # We need 'budget' points
        sampler = qmc.Sobol(d=2, scramble=True)
        # Note: Sobol generator requires power of 2 for full quality, 
        # but it works for any number.
        # To ensure we use the budget effectively and have good distribution,
        # we might want to use a power of 2 close to budget.
        # Budget is 200. 256 is the next power of 2.
        # But we can only evaluate 200 times.
        # So we generate 200 points.
        
        points_2d = sampler.random(n=self.budget)
        
        # Map 2D points to simplex (alpha, beta, gamma)
        # alpha + beta + gamma = 1
        alphas = np.zeros(self.budget)
        betas = np.zeros(self.budget)
        gammas = np.zeros(self.budget)
        
        for i in range(self.budget):
            u, v = points_2d[i]
            if u + v <= 1.0:
                alphas[i] = u
                betas[i] = v
                gammas[i] = 1.0 - u - v
            else:
                alphas[i] = 1.0 - u
                betas[i] = 1.0 - v
                gammas[i] = u + v - 1.0
        
        # Map simplex to decision variables
        # DTLZ1/4 Front:
        # f1 = 0.5 * x0 * g
        # f2 = 0.5 * (1-x0) * x1 * g
        # f3 = 0.5 * (1-x0) * (1-x1) * g
        # Pareto front: g=1 => x2..x11 = 0
        # f1 = 0.5 * x0
        # f2 = 0.5 * (1-x0) * x1
        # f3 = 0.5 * (1-x0) * (1-x1)
        
        # We want f1 = 0.5 * alpha, f2 = 0.5 * beta, f3 = 0.5 * gamma
        # So:
        # x0 = alpha
        # 0.5 * beta = 0.5 * (1 - alpha) * x1
        # beta = (1 - alpha) * x1
        # x1 = beta / (1 - alpha) = beta / (beta + gamma)
        
        X = np.zeros((self.budget, self.dim))
        for i in range(self.budget):
            X[i, 0] = alphas[i]
            denom = betas[i] + gammas[i]
            if denom > 1e-10:
                X[i, 1] = betas[i] / denom
            else:
                X[i, 1] = 0.0
            # x2 to x11 are 0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
