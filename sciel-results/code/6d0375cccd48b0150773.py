"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with constant 0.5
    h = np.full(N, 0.5)
    
    def compute_max_overlap(h):
        """Compute max over shifts of sum_i h[i]*(1-h[(i+s)%N])"""
        # Use FFT for efficient autocorrelation
        # A[s] = sum_i h[i]*h[(i+s)%N] = circular cross-correlation
        H = np.fft.fft(h)
        # A[s] = ifft(fft(h) * conj(fft(h)))... actually circular autocorrelation
        # corr[s] = sum_i h[i] * h[(i+s) % N]
        # This is the circular cross-correlation of h with h
        # Using FFT: corr = ifft(fft(h) * conj(fft(h))) gives sum_i h[i] h[(i-s)%N]
        # We want sum_i h[i] h[(i+s)%N] which is corr[-s] = corr[N-s]
        power = np.abs(H)**2
        A = np.real(np.fft.ifft(power))
        # A[s] = sum_i h[i] * h[(i-s)%N], so A[(-s)%N] = sum_i h[i]*h[(i+s)%N]
        # So the overlap C[s] = N/2 - A[(-s)%N]
        # Max overlap = max_s (N/2 - A[(-s)%N]) = N/2 - min_s A[(-s)%N] = N/2 - min(A)
        return N / 2.0 - np.min(A)
    
    best_score = compute_max_overlap(h)
    
    # Try spectral parameterization: h[i] = 0.5 + sum_{k=1}^{K} (a_k cos(2πki/N) + b_k sin(2πki/N))
    # Optimize to maximize min(A) = minimize max overlap
    
    K = 5  # number of frequency modes
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a = params[:K]
        b = params[K:2*K]
        
        i = np.arange(N)
        h_local = np.full(N, 0.5)
        for k in range(1, K+1):
            h_local += a[k-1] * np.cos(2*np.pi*k*i/N)
            h_local += b[k-1] * np.sin(2*np.pi*k*i/N)
        
        # Clamp to [0,1] - but this breaks differentiability. Use soft penalty instead.
        # Actually, let's use a penalty for out-of-range
        penalty = 0.0
        if np.any(h_local < 0):
            penalty += 1000 * np.sum(np.clip(-h_local, 0, None)**2)
        if np.any(h_local > 1):
            penalty += 1000 * np.sum(np.clip(h_local - 1, 0, None)**2)
        
        # Also need sum(h) = N/2. The DC component is fixed at 0.5, so sum = N/2 automatically
        # since sin and cos have zero mean.
        
        H = np.fft.fft(h_local)
        power = np.abs(H)**2
        A = np.real(np.fft.ifft(power))
        
        score = N / 2.0 - np.min(A) + penalty
        return score
    
    # Try several random initializations
    best_params = None
    best_obj = best_score
    
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            x0 = np.zeros(2*K)
        else:
            x0 = np.random.randn(2*K) * 0.1
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if time.time() > deadline:
                break
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        a = best_params[:K]
        b = best_params[K:2*K]
        i = np.arange(N)
        h_new = np.full(N, 0.5)
        for k in range(1, K+1):
            h_new += a[k-1] * np.cos(2*np.pi*k*i/N)
            h_new += b[k-1] * np.sin(2*np.pi*k*i/N)
        
        # Clamp to [0,1] and renormalize sum to N/2
        h_new = np.clip(h_new, 0, 1)
        current_sum = np.sum(h_new)
        if current_sum > 0:
            h_new = h_new * (N / 2.0) / current_sum
            h_new = np.clip(h_new, 0, 1)
            # Final adjustment to ensure exact sum
            diff = N/2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            # Ensure exact sum by adjusting one element
            diff = N/2.0 - np.sum(h_new)
            if abs(diff) > 1e-10:
                idx = np.argmin(np.abs(h_new - 0.5))
                h_new[idx] += diff
                h_new[idx] = np.clip(h_new[idx], 0, 1)
                # Re-adjust if clipping broke sum
                diff2 = N/2.0 - np.sum(h_new)
                if abs(diff2) > 1e-10:
                    # Distribute among multiple elements
                    for j in range(N):
                        if abs(diff2) < 1e-10:
                            break
                        if diff2 > 0 and h_new[j] < 1.0:
                            adj = min(diff2, 1.0 - h_new[j])
                            h_new[j] += adj
                            diff2 -= adj
                        elif diff2 < 0 and h_new[j] > 0.0:
                            adj = max(diff2, -h_new[j])
                            h_new[j] += adj
                            diff2 -= adj
    
    # Verify and return
    h = np.clip(h, 0, 1)
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if abs(s - N/2.0) > 1e-10:
        diff = N/2.0 - s
        # Distribute difference
        for j in range(N):
            if abs(diff) < 1e-10:
                break
            if diff > 0 and h[j] < 1.0:
                adj = min(diff, 1.0 - h[j])
                h[j] += adj
                diff -= adj
            elif diff < 0 and h[j] > 0.0:
                adj = max(diff, -h[j])
                h[j] += adj
                diff -= adj
    
    return h
# EVOLVE-BLOCK-END
