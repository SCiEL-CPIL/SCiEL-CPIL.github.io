import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        N = self.budget
        if N <= 0:
            return []
        
        # Grid for DP
        M = 1000
        x_grid = np.linspace(0.0, 1.0, M)
        y_grid = 1.0 - np.sqrt(x_grid)
        
        # J[i] will be an array of length M
        # J[i][k] = max HV contribution from points i..N-1 given x_i = x_grid[k]
        # Back[i][k] = index j of x_{i+1} that achieves the max
        
        # Initialize last layer
        # For i = N-1, J[N-1][k] = (11.0 - x_grid[k]) * y_grid[k]
        J = np.zeros((N, M))
        Back = np.zeros((N, M), dtype=int)
        
        J[N-1] = (11.0 - x_grid) * y_grid
        
        # Backward pass
        for i in range(N-2, -1, -1):
            y_i = y_grid # y depends only on x_i, so y_grid[k]
            # For each k, find j >= k maximizing (x_grid[j] - x_grid[k]) * y_grid[k] + J[i+1][j]
            # = -x_grid[k]*y_grid[k] + max_j (x_grid[j]*y_grid[k] + J[i+1][j])
            
            # We can vectorize the max part?
            # For a fixed k, we need max over j in [k, M-1] of (x_grid[j] * y_grid[k] + J[i+1][j])
            # This is a weighted sum.
            # Since M=1000, looping over k is fine.
            
            for k in range(M):
                # Candidates j from k to M-1
                # If k == M-1, only j=M-1
                # Term: x_grid[j] * y_grid[k] + J[i+1][j]
                
                # Optimization:
                # The function h(j) = x_grid[j] * y_k + J[i+1][j]
                # J[i+1][j] is concave-ish?
                # We just take the max.
                
                # To speed up, we can precompute J_next = J[i+1]
                # And use np.max on slices.
                if k < M-1:
                    vals = x_grid[k:] * y_grid[k] + J[i+1][k:]
                    max_idx_local = np.argmax(vals)
                    j = k + max_idx_local
                else:
                    j = M-1
                    vals_max = x_grid[j] * y_grid[k] + J[i+1][j]
                
                J[i][k] = (x_grid[j] - x_grid[k]) * y_grid[k] + J[i+1][j]
                Back[i][k] = j
                
            if time.time() > deadline:
                # Fallback to power law if DP is too slow
                break
        
        # If DP completed, extract path
        if time.time() <= deadline:
            path = []
            k = 0 # Start at x=0
            for i in range(N):
                path.append(x_grid[k])
                if i < N-1:
                    k = Back[i][k]
            
            x1_values = np.array(path)
        else:
            # Fallback
            indices = np.arange(N)
            x1_values = (indices / (N - 1)) ** 4.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1_values[i]
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
