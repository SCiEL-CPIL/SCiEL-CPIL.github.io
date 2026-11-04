import numpy as np
from scipy.optimize import minimize_scalar

N = 14

def construct():
    # Initial guess: uniform distribution
    deltas = np.full(N, 2 * np.pi / N)
    
    def calc_ratio(d):
        # d: array of N gaps
        if np.any(d <= 0) or abs(np.sum(d) - 2 * np.pi) > 1e-8:
            return 0.0
        
        phi = np.zeros(N)
        phi[1:] = np.cumsum(d[:N-1])
        
        min_num = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        den = np.sum(np.sin(d))
        if den <= 0:
            return 0.0
            
        return min_num / den

    # Coordinate Descent
    # We optimize deltas[0..N-2], deltas[N-1] is determined by sum constraint
    
    for iteration in range(200):
        improved = False
        for i in range(N-1):
            # Optimize delta[i]
            # We fix all other deltas.
            # Let current sum of other deltas be S_other.
            # delta[i] + delta[N-1] = 2*pi - S_other
            # Let delta[N-1] = L - delta[i]
            # We need delta[i] > 0 and delta[N-1] > 0 => 0 < delta[i] < L
            
            S_other = np.sum(deltas) - deltas[i] - deltas[N-1]
            L = 2 * np.pi - S_other
            
            if L <= 1e-6:
                continue
                
            def obj_1d(x):
                new_deltas = deltas.copy()
                new_deltas[i] = x
                new_deltas[N-1] = L - x
                if new_deltas[N-1] <= 0 or x <= 0:
                    return 1.0 # Return large positive value for minimization of -ratio
                r = calc_ratio(new_deltas)
                return -r
            
            res = minimize_scalar(obj_1d, bounds=(1e-6, L - 1e-6), method='bounded', options={'xatol': 1e-9})
            
            if res.success:
                new_val = -res.fun
                curr_val = calc_ratio(deltas)
                if new_val > curr_val + 1e-12:
                    deltas[i] = res.x
                    deltas[N-1] = L - res.x
                    improved = True
        
        if not improved:
            break
            
    # Final construction
    angles = np.zeros(N)
    angles[1:] = np.cumsum(deltas[:N-1])
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
