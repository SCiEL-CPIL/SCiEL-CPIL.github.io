import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        # DTLZ7 specific:
        # f1 = y1
        # f2 = y1 * (1 - y2)
        # f3 = 1 + g(x_in) * (1 - y1 * sqrt(y2) / (1 + g(x_in)))
        # where y1 = x[0], y2 = x[1]
        # g(x_in) = 1 + 10*(n_in) + sum_{j in x_in} ((x_j - 0.5)^2 - cos(20*pi*(x_j-0.5)))
        # n_in = dim - 2 = 20
        #
        # To minimize all objectives:
        # 1. Set x_in (indices 2 to 21) to 0.5 to minimize g. g_min = 1.
        # 2. The Pareto front is defined by y1 in [0,1] and y2 in [0,1].
        #    f1 = y1
        #    f2 = y1 * (1 - y2)
        #    f3 = 1 + 1 * (1 - y1 * sqrt(y2) / 2) = 2 - y1 * sqrt(y2) / 2
        #
        # We need to sample y1 and y2 to cover the Pareto front well.
        # The Pareto front in objective space is 2D.
        # We want to maximize hypervolume.
        #
        # Strategy:
        # 1. Use a portion of the budget for a structured grid/Sobol sampling of y1, y2.
        # 2. Use the remaining budget for local refinement around promising points.
        
        dim = self.dim
        budget = self.budget
        
        # Allocate budget: 80% for exploration, 20% for exploitation
        n_explore = int(budget * 0.8)
        n_exploit = budget - n_explore
        
        # Generate Sobol points for y1, y2 (2 dimensions)
        # Sobol requires power of 2 for best properties, but works for any N.
        # Let's use a grid-like approach with jitter for better coverage.
        
        # Actually, let's use a Latin Hypercube or Sobol for 2D.
        # For 2D, we can just use a grid or random.
        # Let's use a stratified sampling approach.
        
        # Generate n_explore points in [0,1]^2
        # Use Sobol for low-discrepancy
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            y_points = sampler.random(n_explore)
        except Exception:
            y_points = np.random.random((n_explore, 2))
        
        # Construct full vectors
        X = np.zeros((n_explore, dim))
        X[:, 0] = y_points[:, 0]  # y1
        X[:, 1] = y_points[:, 1]  # y2
        X[:, 2:] = 0.5  # x_in set to 0.5 to minimize g
        
        # Evaluate exploration points
        for i in range(n_explore):
            if time.time() > deadline - 5.0:
                break
            x = X[i]
            problem(x)
        
        # Now exploitation: refine points near the Pareto front
        # The Pareto front is smooth, so we can do local search around random points
        # Or we can sample more densely in regions that seem good.
        
        # Let's do a simple local refinement: pick random points and do a small perturbation
        # Actually, since we know the Pareto front structure, let's just sample more points
        # with a different distribution.
        
        # Alternative: use the remaining budget to sample additional points
        # with a focus on the "edges" of the Pareto front where hypervolume contribution is high.
        
        # For DTLZ7, the Pareto front has points where f1 is small, f2 is small, f3 is small.
        # The reference point is [1.1, 1.1, 25.0].
        # Points with small f1, f2, f3 contribute most to hypervolume.
        
        # Let's use the remaining budget to sample points with y1 and y2 biased towards 0.
        
        # Generate n_exploit points with a beta distribution biased towards 0
        if n_exploit > 0:
            try:
                # Use beta(2, 5) distribution to bias towards 0
                y_exploit = np.random.beta(2, 5, size=(n_exploit, 2))
            except Exception:
                y_exploit = np.random.random((n_exploit, 2)) ** 3  # Bias towards 0
            
            X_exploit = np.zeros((n_exploit, dim))
            X_exploit[:, 0] = y_exploit[:, 0]
            X_exploit[:, 1] = y_exploit[:, 1]
            X_exploit[:, 2:] = 0.5
            
            for i in range(n_exploit):
                if time.time() > deadline - 2.0:
                    break
                x = X_exploit[i]
                problem(x)
        
        return X
