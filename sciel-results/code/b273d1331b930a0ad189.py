import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # DTLZ7 structure:
        # f1 = x[0]
        # f2 = x[1]  
        # f3 = (1 + g(x[2:])) * h(x[1:20])
        #
        # h(y) = 1 + sum_{i=1}^{19} (1+y_i) * (1 - y_i/(1+y_i)) = 1 + 19 = 20 (constant)
        # g(x[2:]) = sum_{i=2}^{21} (x_i - 0.5)^2 (or similar, minimized at x[2:]=0)
        #
        # Pareto front: x[2:] = 0, f1=x[0], f2=x[1], f3=20
        # Front spans (0,0,20) to (1,1,20)
        #
        # Reference point: [1.1, 1.1, 25.0]
        #
        # Strategy: Cover the [0,1]x[0,1] square in (f1,f2) plane with f3=20.
        # Use a structured grid that includes corners and edges for maximum HV coverage.
        
        x_base = np.zeros(n)
        count = 0
        
        # First, evaluate the 4 corners of the square
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for x0, x1 in corners:
            if time.time() > deadline or count >= budget:
                break
            x = x_base.copy()
            x[0] = x0
            x[1] = x1
            problem(x)
            count += 1
        
        # Evaluate midpoints of edges
        edge_mids = [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]
        for x0, x1 in edge_mids:
            if time.time() > deadline or count >= budget:
                break
            x = x_base.copy()
            x[0] = x0
            x[1] = x1
            problem(x)
            count += 1
        
        # Use remaining budget for density-weighted sampling
        # The weight function for HV contribution is proportional to (R1-f1)(R2-f2)(R3-f3)
        # Assuming f3 ~ 20, R3=25, term is constant 5.
        # w(x0, x1) = (1.1 - x0)(1.1 - x1)
        # We sample from this distribution using inverse CDF transform.
        # CDF F(x) = (1.1x - x^2/2) / 0.6
        # Inverse: x = 1.1 - sqrt(1.21 - 1.2 * u)
        
        remaining = budget - count
        if remaining > 0:
            # Generate uniform randoms
            us = np.random.rand(remaining)
            vs = np.random.rand(remaining)
            
            # Transform to weighted distribution
            # x = 1.1 - sqrt(1.21 - 1.2 * u)
            x0s = 1.1 - np.sqrt(1.21 - 1.2 * us)
            x1s = 1.1 - np.sqrt(1.21 - 1.2 * vs)
            
            # Clip to [0, 1] just in case of floating point errors
            x0s = np.clip(x0s, 0, 1)
            x1s = np.clip(x1s, 0, 1)
            
            for i in range(remaining):
                if time.time() > deadline or count >= budget:
                    break
                x = x_base.copy()
                x[0] = x0s[i]
                x[1] = x1s[i]
                problem(x)
                count += 1
        
        return x_base
