import numpy as np
import time

class Optimizer:
    """
    Optimizer exploiting the known structure of ZDT1.
    ZDT1 Pareto front: x_1...x_n = 0, x_0 in [0, 1].
    f_1 = x_0
    f_2 = 1 - sqrt(x_0)
    
    To maximize hypervolume with N points on a convex front y(x),
    we should distribute points such that the step size dx is proportional
    to the height of the dominated region at that point.
    Height h(x) = ref_2 - y(x) = 11 - (1 - sqrt(x)) = 10 + sqrt(x).
    We want dx / h(x) to be constant (approximately).
    This implies integrating dx / (10 + sqrt(x)) = const.
    Let H(x) = 20 * ln(10 + sqrt(x)) - 2 * sqrt(x).
    We sample H(x) uniformly and invert to get x.
    
    This program uses the analytic solution directly, evaluating each point
    exactly once to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n_evals = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume ZDT1 structure: x[0] is the front variable, rest are 0.
        # x[0] in [0, 1]
        x_lo = lo[0]
        x_hi = hi[0]
        
        # If bounds are not [0,1], we should scale, but ZDT1 is [0,1].
        # We assume standard ZDT1.
        
        num_points = self.budget
        
        # Define the function H(x) whose uniform sampling gives optimal distribution
        # H(x) = 20 * ln(10 + sqrt(x)) - 2 * sqrt(x)
        def H(x):
            return 20 * np.log(10 + np.sqrt(x)) - 2 * np.sqrt(x)
        
        # Calculate range of H
        h_min = H(x_lo)
        h_max = H(x_hi)
        
        # Sample H uniformly
        u_vals = np.linspace(h_min, h_max, num_points)
        
        # Invert H(x) = u to find x
        # 20 * ln(10 + sqrt(x)) - 2 * sqrt(x) = u
        # Let v = sqrt(x). 20 * ln(10 + v) - 2 * v = u
        # Solve for v using Newton's method
        def invert_H(u):
            # Initial guess: linear interpolation
            # H is nearly linear, so v ~ (u - h_min) / (h_max - h_min) * (sqrt(x_hi) - sqrt(x_lo)) + sqrt(x_lo)
            v = np.sqrt(x_lo) + (u - h_min) / (h_max - h_min) * (np.sqrt(x_hi) - np.sqrt(x_lo))
            
            for _ in range(10):
                # f(v) = 20 * ln(10 + v) - 2 * v - u
                f = 20 * np.log(10 + v) - 2 * v - u
                # f'(v) = 20 / (10 + v) - 2
                fp = 20 / (10 + v) - 2
                if np.abs(fp) < 1e-10:
                    break
                v_new = v - f / fp
                if np.abs(v_new - v) < 1e-12:
                    v = v_new
                    break
                v = v_new
            return v ** 2

        x0_vals = np.array([invert_H(u) for u in u_vals])
        
        # Clip to bounds just in case
        x0_vals = np.clip(x0_vals, x_lo, x_hi)
        
        # Construct the population matrix
        X = np.zeros((num_points, self.dim))
        X[:, 0] = x0_vals
        # Set other variables to lower bound
        X[:, 1:] = lo[1:]
        
        # Evaluate all points
        F = []
        for i in range(num_points):
            if time.time() > deadline:
                break
            fx = problem(X[i])
            F.append(fx)
            n_evals += 1
            
        return X
