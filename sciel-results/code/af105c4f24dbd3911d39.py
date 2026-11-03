import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 150.0
    
    def compute_autocorr_fft(h):
        # Use FFT for circular autocorrelation
        # ac[k] = sum_i h[i] * h[(i+k) % N]
        # This is the circular cross-correlation of h with itself
        n = len(h)
        # Zero-pad to avoid circular convolution issues (though we want circular)
        # For circular autocorrelation: ifft(fft(h) * conj(fft(h)))
        H = np.fft.fft(h)
        ac = np.real(np.fft.ifft(H * np.conj(H)))
        # Reorder: ac[k] should be sum h[i]*h[(i+k)%N]
        # ifft gives ac[0], ac[N-1], ac[N-2], ... for lags 0, 1, 2, ...
        # Actually: ifft(|H|^2)[k] = sum_j h[j] * h[(j-k) mod N] (conventional)
        # We want sum_j h[j] * h[(j+k) mod N] = ac_circ[k]
        # ac_circ[k] = ifft(|H|^2)[N-k] for k>0, ac_circ[0] = ifft(|H|^2)[0]
        ac_reordered = np.zeros(n)
        ac_reordered[0] = ac[0]
        for k in range(1, n):
            ac_reordered[k] = ac[n-k]
        return ac_reordered
    
    def compute_autocorr_direct(h):
        ac = np.zeros(N)
        for k in range(N):
            ac[k] = np.dot(h, np.roll(h, -k))
        return ac
    
    def smooth_max(ac, tau):
        # Log-sum-exp for numerical stability
        m = np.max(ac)
        if m < -1000:
            return m
        return m + (1.0/tau) * np.log(np.sum(np.exp(tau * (ac - m))))
    
    def objective(h_flat, tau=20.0, penalty_weight=5000.0):
        h = h_flat.reshape(N)
        sum_err = np.sum(h) - N/2
        ac = compute_autocorr_fft(h)
        smax = smooth_max(ac, tau)
        return smax + penalty_weight * sum_err**2
    
    from scipy.optimize import minimize
    
    best_h = None
    best_score = np.inf
    
    # Multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Different initializations
        if r == 0:
            # Binary initialization
            h = np.zeros(N)
            indices = np.random.permutation(N)
            h[indices[:N//2]] = 1.0
            x0 = h.copy()
        elif r == 1:
            # Uniform initialization
            x0 = np.full(N, 0.5)
        elif r == 2:
            # Random continuous
            x0 = np.random.rand(N) * 0.8 + 0.1
            x0 = x0 / np.sum(x0) * (N/2)
        elif r == 3:
            # Alternating pattern
            x0 = np.zeros(N)
            x0[::2] = 1.0
            x0[1::2] = 0.0
            # Adjust to get sum = N/2
            if N % 2 == 0:
                pass  # Already sum = N/2
            else:
                x0[0] = 0.5
                x0[1] = 0.5
        else:
            # Another random
            x0 = np.random.rand(N)
            x0 = x0 / np.sum(x0) * (N/2)
        
        x0 = np.clip(x0, 0, 1)
        # Ensure sum is close to N/2
        diff = N/2 - np.sum(x0)
        if abs(diff) > 1e-6:
            x0 += diff / N
            x0 = np.clip(x0, 0, 1)
        
        bounds = [(0, 1) for _ in range(N)]
        
        # First pass with moderate tau
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       args=(15.0, 5000.0),
                       options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() - start_time > time_limit - 20:
            if res.fun < best_score:
                best_score = res.fun
                best_h = res.x
            continue
        
        h_candidate = res.x
        
        # Second pass: polish with higher tau and tighter tolerance
        def objective_polish(h_flat, tau=50.0, penalty_weight=10000.0):
            h = h_flat.reshape(N)
            sum_err = np.sum(h) - N/2
            ac = compute_autocorr_fft(h)
            smax = smooth_max(ac, tau)
            return smax + penalty_weight * sum_err**2
        
        res2 = minimize(objective_polish, h_candidate, method='L-BFGS-B', bounds=bounds,
                        args=(50.0, 10000.0),
                        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res2.fun < res.fun:
            h_candidate = res2.x
        
        # Final normalization
        h_candidate = np.clip(h_candidate, 0, 1)
        for _ in range(20):
            diff = N/2 - np.sum(h_candidate)
            if abs(diff) < 1e-10:
                break
            # Distribute to elements with room
            if diff > 0:
                # Need to increase sum; add to elements < 1
                mask = h_candidate < 1.0 - 1e-10
                if np.sum(mask) > 0:
                    h_candidate[mask] += diff / np.sum(mask)
                else:
                    break
            else:
                # Need to decrease sum; subtract from elements > 0
                mask = h_candidate > 1e-10
                if np.sum(mask) > 0:
                    h_candidate[mask] += diff / np.sum(mask)
                else:
                    break
            h_candidate = np.clip(h_candidate, 0, 1)
        
        # Evaluate actual score (max autocorrelation)
        ac_final = compute_autocorr_direct(h_candidate)
        actual_score = np.max(ac_final)
        
        if actual_score < best_score:
            best_score = actual_score
            best_h = h_candidate.copy()
    
    if best_h is None:
        # Fallback: uniform
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
