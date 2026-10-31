import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Objective: minimize max_i h_i * (1 - h_{i+1})
    # where h_{N} = 0 (zero-padded, not circular)
    
    def score(h):
        # h has length N, h_{i+1} for i=0..N-2 is h[i+1], h_N = 0
        overlaps = h[:-1] * (1 - h[1:])
        # Last element: h[N-1] * (1 - 0) = h[N-1]
        overlaps = np.append(overlaps, h[-1])
        return np.max(overlaps)
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    
    # Use scipy to minimize the smooth max approximation
    # Or directly minimize the max with a penalty approach
    
    # Let's try: minimize t + lambda * (sum(h) - N/2)^2 + penalties for bounds
    # Actually, let's use a direct approach with scipy
    
    def objective(x):
        h = x
        overlaps = h[:-1] * (1 - h[1:])
        overlaps = np.append(overlaps, h[-1])
        # Smooth max approximation
        t = np.max(overlaps)
        return t
    
    # Constraints: sum(h) = N/2
    # Bounds: 0 <= h_i <= 1
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    result = minimize(objective, h0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
    
    if result.success:
        h_opt = result.x
    else:
        h_opt = h0
    
    # Verify and return
    h_opt = np.clip(h_opt, 0, 1)
    # Ensure sum is exactly N/2
    diff = N/2 - np.sum(h_opt)
    if abs(diff) > 1e-10:
        # Distribute the difference
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
        # Re-normalize
        s = np.sum(h_opt)
        if s > 0:
            h_opt = h_opt * (N/2) / s
            h_opt = np.clip(h_opt, 0, 1)
    
    return h_opt
