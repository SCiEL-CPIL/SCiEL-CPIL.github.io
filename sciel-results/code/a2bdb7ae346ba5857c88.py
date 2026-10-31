import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    # We want to minimize V(h) = max_k sum h_i (1-h_{i-k})
    # Equivalent to minimizing -min_k sum h_i h_{i-k}
    # Let A_k(h) = sum h_i h_{i-k}
    # Objective: -min_k A_k(h)
    
    # SLSQP requires smooth functions. min is not smooth.
    # We can use a penalty method or just minimize the max overlap directly?
    # Max overlap is max of smooth functions, so it is smooth (convex).
    # Minimizing a convex function is easy.
    
    def objective(h):
        # h is flattened
        h = np.asarray(h)
        # Compute overlaps for all k
        # Overlap_k = sum h_i (1 - h_{i-k})
        # We can compute all autocorrelations at once
        # A_k = sum h_i h_{i-k}
        # Using FFT for speed
        f = np.fft.fft(h)
        power = f * np.conj(f)
        A = np.fft.ifft(power).real
        # A[k] is the circular autocorrelation
        # Overlap_k = N/2 - A[k]
        overlaps = N/2 - A
        # We care about k=1..N-1
        # k=0 is self-overlap, usually excluded or handled separately
        # The task says "shifted", so k!=0
        max_overlap = np.max(overlaps[1:])
        return max_overlap

    # Constraints
    # Sum h = N/2
    # 0 <= h <= 1
    
    cons = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
        {'type': 'ineq', 'fun': lambda h: h}, # h >= 0
        {'type': 'ineq', 'fun': lambda h: 1 - h} # h <= 1
    ]
    
    bounds = [(0, 1) for _ in range(N)]
    
    res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
    
    if res.success:
        # Clean up: ensure sum is exactly N/2
        h_opt = res.x
        h_opt = np.clip(h_opt, 0, 1)
        # Adjust for sum
        diff = N/2 - np.sum(h_opt)
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
        return h_opt
    else:
        return h0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    h0 = np.full(N, 0.5)
    
    def objective(h):
        h = np.asarray(h)
        f = np.fft.fft(h)
        power = f * np.conj(f)
        A = np.fft.ifft(power).real
        overlaps = N/2 - A
        # Exclude k=0
        return np.max(overlaps[1:])

    cons = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
        {'type': 'ineq', 'fun': lambda h: h},
        {'type': 'ineq', 'fun': lambda h: 1 - h}
    ]
    
    bounds = [(0, 1) for _ in range(N)]
    
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            h_opt = res.x
            h_opt = np.clip(h_opt, 0, 1)
            # Force sum
            diff = N/2 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            return h_opt
    except Exception:
        pass
    return h0
# EVOLVE-BLOCK-END
