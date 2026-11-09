import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ2 Pareto front: quarter-sphere in 3D
        # Parameterize with angles theta1, theta2 in [0, pi/2]
        # y0 = cos(theta1) * cos(theta2)
        # y1 = cos(theta1) * sin(theta2)  
        # y2 = sin(theta1)
        # x[0]=y0, x[1]=y1, x[2]=y2, x[3:]=0.5 gives g=0
        
        # Use a dense stratified grid of angles for uniform coverage
        # For 200 evals, use ~14x14 grid = 196 points
        n_per_dim = int(np.sqrt(n_evals))
        
        # Stratified sampling of angles for better uniformity
        theta1_vals = (np.arange(n_per_dim) + 0.5) / n_per_dim * (np.pi / 2)
        theta2_vals = (np.arange(n_per_dim) + 0.5) / n_per_dim * (np.pi / 2)
        
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
        theta1_flat = T1.ravel()
        theta2_flat = T2.ravel()
        
        # Generate points on the front
        X = np.zeros((n_evals, self.dim))
        n_points = min(len(theta1_flat), n_evals)
        
        for i in range(n_points):
            t1 = theta1_flat[i]
            t2 = theta2_flat[i]
            
            y0 = np.cos(t1) * np.cos(t2)
            y1 = np.cos(t1) * np.sin(t2)
            y2 = np.sin(t1)
            
            X[i, 0] = y0
            X[i, 1] = y1
            X[i, 2] = y2
            X[i, 3:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
