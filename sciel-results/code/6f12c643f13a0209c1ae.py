import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    best_h = np.ones(N)
    best_val = np.inf
    
    # Objective function: max of autocorrelation
    def objective(h):
        # h is flat array of length N
        C = np.convolve(h, h)
        return np.max(C)
        
    def gradient(h):
        C = np.convolve(h, h)
        k_max = np.argmax(C)
        # Gradient of C[k_max] w.r.t h
        # d/dh[j] sum_i h[i] h[k_max - i] = h[k_max - j] + h[k_max - j] = 2 * h[k_max - j]
        # Valid indices for j: 0 <= j < N
        # Valid indices for k_max - j: 0 <= k_max - j < N => k_max - N < j <= k_max
        grad = np.zeros(N)
        idx = k_max - np.arange(N)
        valid = (idx >= 0) & (idx < N)
        grad[valid] = 2.0 * h[idx[valid]]
        return grad

    # We want to minimize value = abs(2 * N * max(C) / sum(h)^2)
    # By constraining sum(h) = 1, we minimize abs(2 * N * max(C))
    # Since we want to lower the value, and seed is 2.0 (positive), 
    # minimizing max(C) is the goal.
    # Note: If max(C) becomes negative, abs(2*N*max(C)) increases? 
    # Wait. value = abs(2 * N * max(C)).
    # If max(C) is negative, say -1, value is 2N.
    # If max(C) is positive, say 1, value is 2N.
    # We want value to be small.
    # So we want max(C) to be close to 0?
    # But sum(h)=1 implies max(C) >= (sum(h))^2 / N = 1/N by Cauchy-Schwarz?
    # Actually, (sum h)^2 = (sum h_i * 1)^2 <= (sum h_i^2) * N.
    # Also max(h*h) >= (sum h)^2 / (2N-1) roughly?
    # Let's check the seed: h=1, sum=400, max(C)=400.
    # value = 2 * 400 * 400 / 400^2 = 2.
    # If we make h sparse?
    # If h is a delta function at 0: sum=1, max(C)=1.
    # value = 2 * 400 * 1 / 1 = 800. Worse.
    # We want max(C) to be small.
    # The minimum possible max(C) for sum=1 is roughly 1/N if h is uniform?
    # If h is uniform 1/400, C[0]=400*(1/400)^2 = 1/400.
    # value = 2 * 400 * (1/400) / 1 = 2.
    # So uniform gives 2.
    # Can we get lower than 2?
    # The inequality suggests C3 is bounded.
    # If we can make max(C) < 1/N, we get value < 2.
    # But max(C) >= (sum h)^2 / (2N-1) is not necessarily true for max.
    # Actually, for any h, max_k (h*h)_k >= (sum h)^2 / (2N-1) ?
    # Sum of all C[k] = (sum h)^2.
    # There are 2N-1 terms.
    # So max(C) >= (sum h)^2 / (2N-1).
    # For sum=1, max(C) >= 1/799.
    # Then value >= 2 * 400 * (1/799) = 800/799 > 1.
    # So the theoretical lower bound is ~1.
    # We want to approach 1.
    # So we want to minimize max(C).
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    # Try several random initializations
    for _ in range(20):
        # Random initialization, normalized to sum 1
        h0 = np.random.randn(N)
        h0 = h0 / np.sum(h0)
        
        # Optimize
        try:
            res = minimize(
                objective, 
                h0, 
                jac=gradient, 
                method='L-BFGS-B', 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            h_opt = res.x
            # Re-normalize to ensure constraint is met exactly
            S = np.sum(h_opt)
            if np.abs(S) < 1e-12:
                continue
            h_opt = h_opt / S
            
            # Evaluate
            C = np.convolve(h_opt, h_opt)
            M = np.max(C)
            val = abs(2 * N * M)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
    return best_h
