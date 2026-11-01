import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(x):
    """
    Compute the linear autocorrelation of a sequence x.
    Returns array of length 2N-1, where index k corresponds to lag k-(N-1).
    Actually, let's return lags 0 to N-1 for simplicity in metric calculation.
    R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    """
    # Use FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.fft(x, n=n_fft)
    power = np.abs(F) ** 2
    acf = np.fft.ifft(power).real
    
    # The linear autocorrelation lags 0 to N-1 are in acf[0:N]
    # Note: acf[0] is the energy sum x^2.
    return acf[:N]

def _max_sidelobe(x):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes (lags 1 to N-1).
    """
    acf = _compute_autocorr(x)
    # acf[0] is lag 0 (energy), acf[1] is lag 1, etc.
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _objective_and_gradient(x):
    """
    Objective: Sum of squares of the sidelobes (lags 1 to N-1).
    This is a smooth surrogate for the peak sidelobe.
    Returns (value, gradient).
    """
    # Compute autocorrelation using convolution
    # R[k] = sum_{i} x[i] x[i+k]
    # We can compute all R[k] for k=0..N-1 efficiently.
    
    # Gradient of R[k] w.r.t x[i]:
    # d/dx_i (sum_j x[j] x[j+k]) = x[i+k] + x[i-k] (with boundary checks)
    
    # Let's compute R and Grad using loops or vectorized ops.
    # For N=400, a simple loop over lags is fast enough.
    R = np.zeros(N)
    grad = np.zeros(N)
    
    for k in range(N):
        # R[k] = dot(x[:N-k], x[k:])
        R[k] = np.dot(x[:N-k], x[k:])
        
        # Gradient contribution for lag k:
        # d R[k] / d x[i] = x[i+k] if i+k < N else 0
        #                 + x[i-k] if i-k >= 0 else 0
        # This is equivalent to:
        # grad[:N-k] += x[k:]
        # grad[k:] += x[:N-k]
        
        if k == 0:
            grad[:] += 2 * x
        else:
            grad[:N-k] += x[k:]
            grad[k:] += x[:N-k]
            
    # We only care about sidelobes k=1..N-1
    # Objective: sum_{k=1}^{N-1} R[k]^2
    val = np.sum(R[1:] ** 2)
    
    # Gradient: 2 * sum_{k=1}^{N-1} R[k] * d R[k] / d x
    # We already accumulated grad for all k=0..N-1.
    # We need to subtract the contribution of k=0 from grad.
    # grad_total = 2*x (from k=0) + sum_{k=1}^{N-1} (x[k:]+x[:N-k])
    # We want grad_sidelobes = sum_{k=1}^{N-1} (x[k:]+x[:N-k])
    # So grad_sidelobes = grad_total - 2*x
    
    grad_sidelobes = grad - 2 * x
    
    # Final gradient of val = 2 * sum R[k] * grad_R_k
    # Wait, the chain rule: d/dx (sum R[k]^2) = sum 2 R[k] d R[k]/dx
    # My 'grad' variable above is d R_total / dx.
    # Let's re-calculate properly.
    
    # Let's restart the gradient calculation to be safe.
    g = np.zeros(N)
    for k in range(1, N):
        Rk = R[k]
        # d R[k] / dx is x[k:] in first N-k positions and x[:N-k] in last k positions
        g[:N-k] += Rk * x[k:]
        g[k:] += Rk * x[:N-k]
        
    grad_val = 2 * g
    return val, grad_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    # We want to minimize the peak sidelobe.
    # Strategy:
    # 1. Generate random continuous initializations in [-1, 1].
    # 2. Optimize the sum of squares of sidelobes using L-BFGS-B.
    # 3. Round to +/- 1.
    # 4. Perform discrete local search (flipping bits) to minimize the actual peak sidelobe.
    # 5. Keep the best result.
    
    best_h = None
    best_score = float('inf')
    
    # Number of restarts depends on time.
    # Each L-BFGS-B run might take ~1-5 seconds.
    # Discrete search is fast.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random initialization
        # Start with random continuous values
        x0 = np.random.uniform(-1, 1, N)
        
        # Optimization
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                           bounds=[(-1, 1)] * N, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        except Exception:
            continue
            
        # Round to discrete
        h = np.sign(res.x)
        # Ensure no zeros (sign(0) is 0, map to 1 or -1)
        h[h == 0] = 1
        
        # Discrete local search
        # We want to minimize max |R[k]| for k=1..N-1
        current_max = _max_sidelobe(h)
        
        if current_max < best_score:
            best_score = current_max
            best_h = h.copy()
            
        # Local search: try flipping each bit
        # To speed up, we can precompute R or use FFT.
        # For N=400, FFT is fast.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline:
                    break
                # Flip bit i
                old_val = h[i]
                h[i] = -old_val
                new_max = _max_sidelobe(h)
                
                if new_max < current_max - 1e-9:
                    current_max = new_max
                    improved = True
                    if current_max < best_score:
                        best_score = current_max
                        best_h = h.copy()
                else:
                    # Revert
                    h[i] = old_val
                    
        # If we are running out of time, break
        if time.time() > deadline - 2.0:
            break

    # If no result was found (shouldn't happen), return random
    if best_h is None:
        best_h = np.random.choice([1, -1], N)
        
    return best_h
