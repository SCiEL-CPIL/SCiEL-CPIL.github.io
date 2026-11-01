"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_sidelobe_bound(x):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes.
    x is a 1D array of length N.
    We use FFT for efficient computation.
    """
    # Zero-pad to length 2*N to avoid circular convolution effects
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    R = np.fft.ifft(X * np.conj(X)).real
    # The autocorrelation is centered at index N/2? 
    # np.fft.ifft(fft(x)*conj(fft(x))) gives circular autocorrelation.
    # With zero padding to 2N, the linear autocorrelation is in R[0:N] (lags 0 to N-1) 
    # and R[N+1:2N] corresponds to negative lags.
    # Actually, linear autocorrelation r[k] = sum x[i] x[i+k].
    # Using FFT: r = ifft(fft(x, 2N) * conj(fft(x, 2N)))
    # The result is length 2N. 
    # r[0] is lag 0 (peak).
    # r[1:N] are positive lags 1 to N-1.
    # r[N+1:2N] are negative lags.
    # We want max(|r[k]|) for k != 0.
    
    # Ensure numerical stability
    R = R[:N] # Take first N elements: lags 0 to N-1
    # r[0] is the peak. We exclude it.
    sidelobes = R[1:]
    return np.max(np.abs(sidelobes))

def _objective(x):
    """
    Objective function for the optimizer.
    x is a 1D array of length N (continuous).
    """
    return _compute_sidelobe_bound(x)

def _gradient(x):
    """
    Approximate gradient using finite differences or analytic.
    For speed and robustness with L-BFGS-B, we can let scipy estimate it,
    but providing a gradient is better. 
    However, the max operation makes the gradient non-smooth.
    We will rely on L-BFGS-B's finite difference gradient estimation 
    or use a smoothed objective. 
    Given the 120s limit and N=400, finite difference gradient 
    (400 evaluations per step) might be too slow if many steps are needed.
    Let's try to compute the gradient analytically for the max element.
    
    Let M = max_k |r[k]|.
    dM/dx_i = sign(r[k_max]) * dr[k_max]/dx_i
    dr[k]/dx_i = x[i+k] + x[i-k] (with boundary handling)
    
    This is complex to implement robustly for the "max" switch.
    Let's use the default finite difference for L-BFGS-B but limit iterations.
    Or, better, use a differentiable proxy: sum of squares of sidelobes?
    Minimizing the max is non-smooth. Minimizing the sum of squares (energy) 
    often correlates well and is smooth. 
    However, the task asks for the "upper bound", which is the max.
    
    Let's stick to the max objective but use a small number of L-BFGS-B steps 
    with finite difference gradient. 
    N=400. One gradient eval ~ 400 * (cost of FFT). 
    Cost of FFT ~ N log N. 
    400 * 400 log 400 ~ 1.6e6 ops. Very fast.
    L-BFGS-B can do ~50-100 iterations in 150s easily.
    """
    pass # We won't provide gradient, let scipy estimate.

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # 1. Initialize with random continuous values in [-1, 1]
    # Using multiple restarts to escape local minima
    best_x_cont = None
    best_score = np.inf
    
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 10.0:
            break
            
        # Random initialization
        x0 = np.random.uniform(-1, 1, N)
        
        # Optimization
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            x_opt = res.x
            score = _objective(x_opt)
            
            if score < best_score:
                best_score = score
                best_x_cont = x_opt.copy()
                
        except Exception:
            continue

    if best_x_cont is None:
        # Fallback to seed
        return np.ones(N)

    # 2. Discretize to "signed step function"
    # The prompt implies "signed step heights". 
    # If the domain is {-1, 1}, we threshold at 0.
    # If the domain allows other heights, we might need to scale.
    # Given "signed step function" in the context of autocorrelation inequalities 
    # (like Barker sequences), the standard is {-1, 1}.
    # Let's assume the output should be in {-1, 1}.
    x_discrete = np.sign(best_x_cont)
    # Handle zeros (unlikely with continuous random init)
    x_discrete[x_discrete == 0] = 1.0
    
    # 3. Optional: Local discrete refinement (Flip bits)
    # Check if flipping any bit improves the score.
    # This is a fast O(N^2) or O(N log N) check if done carefully, 
    # but simple loop is O(N^2) which is 160,000 ops, very fast.
    
    def _score_discrete(x):
        return _compute_sidelobe_bound(x)
    
    current_score = _score_discrete(x_discrete)
    improved = True
    while improved and (time.time() - start_time < deadline - 5.0):
        improved = False
        for i in range(N):
            if time.time() - start_time > deadline - 5.0:
                break
            # Try flipping bit i
            x_test = x_discrete.copy()
            x_test[i] = -x_test[i]
            new_score = _score_discrete(x_test)
            if new_score < current_score:
                x_discrete = x_test
                current_score = new_score
                improved = True
                break # Restart scan from 0 as landscape changed
    
    return x_discrete

# EVOLVE-BLOCK-END
