import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap(h):
    """
    Compute the maximum overlap C_k for linear shifts k=1..N-1.
    C_k = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    """
    max_c = 0.0
    for k in range(1, N):
        c_k = np.dot(h[:N-k], 1 - h[k:])
        if c_k > max_c:
            max_c = c_k
    return max_c

def _generate_h(params):
    """
    Generate h from parameters [A1, phi1, A2, phi2, ...]
    h_i = 0.5 + sum_k A_k * cos(2*pi*k*i/N + phi_k)
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    n_params = len(params)
    n_freqs = n_params // 2
    
    for j in range(n_freqs):
        A = params[2*j]
        phi = params[2*j+1]
        freq = j + 1
        h += A * np.cos(2 * np.pi * freq * i / N + phi)
        
    return h

def _objective(params):
    """
    Objective function for optimizer.
    Returns max overlap + penalty for bounds violation.
    """
    h = _generate_h(params)
    
    # Check bounds
    if np.any(h < 0) or np.any(h > 1):
        # Penalty proportional to violation
        viol = np.sum(np.clip(-h, 0, None)) + np.sum(np.clip(h - 1, 0, None))
        return 10.0 + viol * 100.0
        
    # Check sum (should be exactly N/2 due to cosine symmetry if N is even and freqs are integers)
    # Sum of cos(2*pi*k*i/N) over i=0..N-1 is 0 for k not multiple of N.
    # So sum(h) = N*0.5 = N/2.
    
    return _max_overlap(h)

def construct():
    start_time = time.time()
    deadline = 110.0  # seconds
    
    # We will optimize a small number of frequency components.
    # Let's try 3 components.
    n_freqs = 3
    n_params = 2 * n_freqs
    
    # Initial guess:
    # Dominant low frequency.
    # A1 = 0.5, phi1 = 0
    # A2 = 0.1, phi2 = 0
    # A3 = 0.05, phi3 = 0
    best_params = np.array([0.5, 0.0, 0.1, 0.0, 0.05, 0.0])
    best_score = _objective(best_params)
    
    # Local search using Nelder-Mead
    # We do multiple restarts with random initializations to escape local minima.
    
    np.random.seed(42)
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        # Random initialization
        # Amplitudes must be small enough to stay in [0,1].
        # Max possible A is 0.5 if only one freq. With multiple, sum of |A| <= 0.5.
        # Let's sample A such that sum(A) <= 0.5.
        raw_A = np.random.rand(n_freqs) * 0.5
        sum_A = np.sum(raw_A)
        if sum_A > 0.5:
            raw_A *= 0.5 / sum_A
            
        phis = np.random.rand(n_freqs) * 2 * np.pi
        
        params_init = np.zeros(n_params)
        for j in range(n_freqs):
            params_init[2*j] = raw_A[j]
            params_init[2*j+1] = phis[j]
            
        # Optimization
        try:
            res = minimize(_objective, params_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Generate final h
    h = _generate_h(best_params)
    
    # Ensure strict constraints
    # Clip to [0,1]
    h = np.clip(h, 0, 1)
    
    # Adjust sum to be exactly N/2
    # The cosine construction should yield sum N/2 exactly (floating point errors aside).
    # But clipping might break the sum.
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-9:
        # Adjust elements not at bounds
        mask = (h > 1e-6) & (h < 1 - 1e-6)
        if np.any(mask):
            idxs = np.where(mask)[0]
            # Distribute diff evenly among adjustable elements
            step = diff / len(idxs)
            for idx in idxs:
                h[idx] += step
                h[idx] = np.clip(h[idx], 0, 1)
            # If still off, adjust one element
            residual = target_sum - np.sum(h)
            if abs(residual) > 1e-9:
                # Find an element with most room
                # Just pick first adjustable
                idx = idxs[0]
                h[idx] += residual
                h[idx] = np.clip(h[idx], 0, 1)
                
    return h
