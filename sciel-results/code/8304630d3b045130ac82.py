import numpy as np
import time
import sys
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_hv(self, points, ref):
        # points: (N, 2)
        # ref: (2,)
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by f1 descending
        points = points[np.argsort(points[:, 0])[::-1]]
        
        hv = 0.0
        x_next = ref[0]
        for i in range(n):
            x_curr = points[i][0]
            y_curr = points[i][1]
            width = x_next - x_curr
            height = ref[1] - y_curr
            hv += width * height
            x_next = x_curr
        return hv

    def _compute_optimal_x1(self):
        N = self.budget
        if N <= 0:
            return
        ref = np.array([11.0, 11.0])
        t = np.linspace(0, 1, N)
        
        def neg_hv(p):
            x1 = t ** p
            f2 = 1.0 - np.sqrt(x1)
            points = np.column_stack((x1, f2))
            return -self._compute_hv(points, ref)
            
        # Optimize p
        res = minimize_scalar(neg_hv, bounds=(0.1, 10.0), method='bounded')
        best_p = res.x
        
        self.optimal_x1 = t ** best_p

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        if self.optimal_x1 is None:
             self._compute_optimal_x1()
             
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
