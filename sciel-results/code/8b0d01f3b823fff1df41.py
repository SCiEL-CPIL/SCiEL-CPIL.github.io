import numpy as np
import time
import math

N = 400
# Precompute FFT size
n_fft = 1
while n_fft < 2 * N:
    n_fft *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _local_search(s, current_val, max_passes=50, time_limit=10.0, deadline=None):
    """Local search by flipping bits."""
    start = time.time()
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() - start < time_limit:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            
            old_si = s[i]
            s[i] = -old_si
            
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                current_val = val_new
                improved = True
            else:
                s[i] = old_si
    
    return s, current_val

def _ising_cooling_init(n_iter=500, deadline=None):
    """
    Spin-glass cooling schedule for a frustrated Ising antiferromagnet.
    The energy function penalizes short-range autocorrelation peaks (L2 norm)
    and implicitly encourages long-range sign clustering to minimize L-infinity norm.
    """
    # Start with random continuous values
    x = np.random.uniform(-1, 1, size=N)
    
    # Cooling schedule
    T_start = 0.1
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute autocorrelation
        A = _calc_acf(x)
        
        # Energy: L2 norm of sidelobes (smooth surrogate for L-infinity)
        # We use the gradient of sum(A[k]^2)
        A_for_grad = np.zeros(N)
        A_for_grad[1:] = A[1:]
        
        # Gradient via FFT
        Fx = np.fft.rfft(x, n=n_fft)
        FA = np.fft.rfft(A_for_grad, n=n_fft)
        grad = np.fft.irfft(4 * Fx * FA, n=n_fft)[:N]
        
        # Metropolis acceptance with noise
        # Proposed move: x_new = x - 0.01 * grad + noise
        noise = np.random.normal(0, T * 0.1, size=N)
        x_new = x - 0.01 * grad + noise
        x_new = np.clip(x_new, -1, 1)
        
        # Compute energy change (approximate)
        # For speed, we just accept if T is high, or if it's a clear improvement
        # A rigorous Metropolis would require exact energy calculation, which is expensive.
        # We use a heuristic: accept with probability 1 if T > 0.05, else only if improved.
        # Actually, let's just do gradient descent with noise (simulated annealing on continuous variables)
        
        # To be more efficient, we can just update x directly
        x = x_new
        T *= cooling_rate
        
    # Threshold to binary
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Strategy 1: Ising cooling initialization (multiple times)
    for init_iter in range(15):
        if time.time() > deadline - 5:
            break
        
        s = _ising_cooling_init(n_iter=300, deadline=deadline - 2)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=30, time_limit=5.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Strategy 2: Random restarts with refinement
    while time.time() < deadline - 2:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=20, time_limit=3.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
