import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Phase 1: Global Initialization with 50 random points
        n_init = 50
        points = rng.uniform(-5, 5, (n_init, self.dim))
        values = []
        for p in points:
            values.append(func(p))
            if time.time() > deadline:
                break
        
        # Sort to get best starting points
        pairs = sorted(zip(values, points), key=lambda x: x[0])
        top_k = min(5, len(pairs))
        best_x = pairs[0][1].copy()
        best_f = pairs[0][0]
        
        # Phase 2: Run (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly
        remaining_budget = self.budget - len(values)
        budget_per_es = max(10, remaining_budget // top_k)
        
        for i in range(top_k):
            if time.time() > deadline:
                break
            
            x = pairs[i][1].copy()
            f = pairs[i][0]
            
            # (1+1)-ES with standard CMA-ES-like parameters
            # sigma starts at 1.0 and adapts
            sigma = 1.0
            c_sigma = 1.0 / (self.dim + 4)  # Adaptation rate for sigma
            c_c = (4 + self.dim / self.dim) / (4 + self.dim + self.dim / self.dim)  # Path adaptation for center
            c_1 = 2 / ((self.dim + 1.3)**2 + self.dim)
            c_mu = min(1 - c_1, 2 * ((self.dim + 1.5) / (self.dim + 3))**2 - 1)
            mu = self.dim // 2 + 1  # Number of best individuals for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu_eff = 1 / np.sum(weights**2)
            chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            
            # State variables
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            C = np.eye(self.dim)
            damps = 1 + 2 * max(0, np.sqrt((mu - 1 + 0.5 / mu) / (self.dim + 2)) - 1) + c_sigma
            
            for step in range(budget_per_es):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                z = rng.standard_normal(self.dim)
                x_trial = x + sigma * (C @ z)
                
                # Clip to box
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                
                if f_trial < f:
                    # Success: update
                    y = (x_trial - x) / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * y
                    x = x_trial
                    f = f_trial
                    
                    if f < best_f:
                        best_f = f
                        best_x = x_trial.copy()
                    
                    # Update C
                    C = (1 - c_1) * C + c_1 * (np.outer(y, y) + (1 - c_sigma) * (2 - c_sigma) * nu_eff * C)
                    
                    # Normalize C to maintain determinant 1
                    det_C = np.linalg.det(C)
                    if det_C > 0:
                        C /= det_C ** (1 / self.dim)
                else:
                    # Failure: no update to x, but still update paths
                    pass
                
                # Adapt sigma
                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_f
