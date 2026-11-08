"""Combined approach: FFT-based overlap + binary sequence search + Fourier optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_j h[j]*(1-h[(j+k)%N]) * (2/N)."""
    n = len(h)
    dx = 2.0 / n
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    sum_h = np.sum(h)
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _binary_autocorr_min(b):
    """For binary b with sum=N/2, compute min of autocorrelation C[k] for k=1..n-1."""
    n = len(b)
    B = np.fft.rfft(b.astype(float), n=2*n)
    autocorr = np.fft.irfft(B * np.conj(B), n=2*n)[:n]
    # C[k] = sum_j b[j]*b[(j+k)%n] = autocorr[(-k)%n]
    # We want min over k=1..n-1 of C[k]
    C_vals = np.zeros(n)
    C_vals[0] = autocorr[0]
    if n > 1:
        C_vals[1:] = autocorr[n-1:0:-1]
    return np.min(C_vals[1:])

def _binary_local_search(b, max_iters=5000):
    """Local search: flip pairs to maintain sum=N/2, accept if min autocorr improves."""
    n = len(b)
    b = b.copy()
    cur_min = _binary_autocorr_min(b)
    ones = np.where(b == 1)[0]
    zeros = np.where(b == 0)[0]
    
    for it in range(max_iters):
        if time.time() > _deadline:
            break
        # Pick random one and zero to swap
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        if i == j:
            continue
        b[i] = 0
        b[j] = 1
        new_min = _binary_autocorr_min(b)
        if new_min > cur_min:
            cur_min = new_min
            ones = np.where(b == 1)[0]
            zeros = np.where(b == 0)[0]
        else:
            # Revert
            b[i] = 1
            b[j] = 0
    
    return b, cur_min

def _objective(coeffs):
    """Objective function for optimization: max overlap from Fourier coefficients."""
    x = np.linspace(0, 2, N, endpoint=False)
    h = np.full(N, 0.5)
    for k in range(len(coeffs) // 2):
        a_k = coeffs[2*k]
        b_k = coeffs[2*k+1]
        h += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    if np.any(h < -1e-8) or np.any(h > 1.0 + 1e-8):
        return 10.0
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    if abs(current_sum - N/2) > 1e-6:
        return 10.0 + abs(current_sum - N/2)
    return _compute_max_overlap(h)

_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # 10s safety margin
    
    # Start with baseline
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Try analytic candidates
    candidates = []
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    candidates.append(np.clip(h1, 0, 1))
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    candidates.append(np.clip(h2, 0, 1))
    h3 = (x < 1.0).astype(float)
    candidates.append(h3)
    
    for h_cand in candidates:
        s = _compute_max_overlap(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Binary sequence search
    np.random.seed(42)
    
    # Initialize balanced binary sequence
    b = np.zeros(N, dtype=int)
    b[:N//2] = 1
    
    # Multiple random restarts for binary search
    n_binary_restarts = 8
    for restart in range(n_binary_restarts):
        if time.time() > _deadline:
            break
        
        if restart == 0:
            # Start with alternating pattern
            b = np.zeros(N, dtype=int)
            b[::2] = 1
            b[1::2] = 0
            # Fix sum to exactly N/2
            s = np.sum(b)
            if s != N//2:
                # Adjust
                if s > N//2:
                    for i in range(N-1, -1, -1):
                        if b[i] == 1 and s > N//2:
                            b[i] = 0
                            s -= 1
                        elif b[i] == 0 and s < N//2:
                            b[i] = 1
                            s += 1
                else:
                    for i in range(N):
                        if b[i] == 0 and s < N//2:
                            b[i] = 1
                            s += 1
        else:
            # Random balanced binary
            b = np.zeros(N, dtype=int)
            idx = np.random.permutation(N)
            b[idx[:N//2]] = 1
        
        b_opt, _ = _binary_local_search(b, max_iters=3000)
        
        h_bin = b_opt.astype(float)
        s = _compute_max_overlap(h_bin)
        if s < best_score:
            best_score = s
            best_h = h_bin.copy()
    
    # Fourier optimization starting from best found
    n_modes = 4
    n_coeffs = 2 * n_modes
    bounds = [(-0.5, 0.5)] * n_coeffs
    
    n_starts = 10
    for start_idx in range(n_starts):
        if time.time() > _deadline:
            break
        
        if start_idx == 0:
            x0 = np.zeros(n_coeffs)
        else:
            x0 = np.random.uniform(-0.3, 0.3, n_coeffs)
        
        try:
            result = minimize(
                _objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                h_new = np.full(N, 0.5)
                for k in range(n_modes):
                    h_new += result.x[2*k] * np.cos(np.pi * k * x)
                    h_new += result.x[2*k+1] * np.sin(np.pi * k * x)
                h_new = np.clip(h_new, 0, 1)
                
                current_sum = np.sum(h_new)
                target_sum = N / 2
                if abs(current_sum - target_sum) > 1e-8:
                    diff = target_sum - current_sum
                    if diff > 0:
                        mask = h_new < 1.0 - 1e-10
                        if np.sum(mask) > 0:
                            h_new[mask] += diff / np.sum(mask)
                    else:
                        mask = h_new > 1e-10
                        if np.sum(mask) > 0:
                            h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
                
                if abs(np.sum(h_new) - N/2) < 1e-6:
                    actual_score = _compute_max_overlap(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
