"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1,
    to minimize the maximum overlap of h with shifted 1-h.
    
    We use a spectral approach: represent h as |F|^2 / sum(|F|^2) where F is a sparse
    complex vector. This enforces non-negativity. We optimize the phases and magnitudes
    of a few non-zero spectral coefficients to minimize the max overlap.
    """
    deadline = time.time() + 110  # 110s safety margin
    
    # The overlap at shift k is sum_i h[i] * (1-h[i+k mod N])
    # We want to minimize max_k overlap_k
    
    # Strategy: use a small number of spectral components.
    # Let h[i] = |sum_{m in M} a_m exp(2*pi*i*m*i/N)|^2 / norm
    # where M is a set of a few frequency indices, a_m are complex amplitudes.
    
    # For simplicity, start with a small number of harmonics.
    # The DC component (m=0) gives the mean. To have sum(h) = N/2, we need the DC part to contribute appropriately.
    
    # Let's use a parametric family: h[i] = 0.5 * (1 + sum_{k=1}^{K} r_k * cos(2*pi*k*i/N + phi_k))
    # This is real, non-negative if the sum of amplitudes is small enough, and has mean 0.5.
    # We can optimize r_k and phi_k to minimize the max overlap.
    
    K = 5  # number of harmonics to use
    
    def params_to_h(params):
        # params: [r_1, phi_1, r_2, phi_2, ..., r_K, phi_K]
        h = np.full(N, 0.5)
        for k in range(1, K+1):
            r = params[2*(k-1)]
            phi = params[2*(k-1)+1]
            h += r * np.cos(2*np.pi*k*np.arange(N)/N + phi)
        # Clip to [0,1] and renormalize to sum = N/2
        h = np.clip(h, 0, 1)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
        # Re-clip after scaling (scaling might push above 1 if we were too aggressive)
        h = np.clip(h, 0, 1)
        # Final adjustment: if sum is not exactly N/2, adjust the largest element
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            idx = np.argmax(h) if diff < 0 else np.argmin(h)
            h[idx] += diff
            h = np.clip(h, 0, 1)
        return h
    
    def overlap_score(h):
        # Compute max overlap: max_k sum_i h[i] * (1-h[i+k])
        # Use FFT for efficiency
        # overlap_k = sum_i h[i] * (1 - h[i+k]) = sum_i h[i] - sum_i h[i]*h[i+k]
        # sum_i h[i] = N/2
        # sum_i h[i]*h[i+k] is the autocorrelation at lag k
        # Autocorrelation via FFT:
        H = np.fft.fft(h)
        auto = np.fft.ifft(np.abs(H)**2).real
        # auto[k] = sum_i h[i] * h[i+k mod N]
        overlaps = N/2.0 - auto
        return np.max(overlaps)
    
    def objective(params):
        h = params_to_h(params)
        return overlap_score(h)
    
    # Initial guess: small random amplitudes
    np.random.seed(42)
    x0 = np.zeros(2*K)
    for k in range(K):
        x0[2*k] = 0.1  # small radius
        x0[2*k+1] = np.random.uniform(0, 2*np.pi)
    
    # Use Nelder-Mead for optimization (no gradient needed, robust)
    best_params = x0.copy()
    best_score = objective(x0)
    
    start_time = time.time()
    
    # Try a few different initializations
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        if trial > 0:
            x0 = np.zeros(2*K)
            for k in range(K):
                x0[2*k] = np.random.uniform(0.01, 0.2)
                x0[2*k+1] = np.random.uniform(0, 2*np.pi)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    h = params_to_h(best_params)
    
    # Final polish: local refinement by adjusting individual elements
    # Try to reduce the max overlap by local moves
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        
        h_curr = h.copy()
        overlaps = N/2.0 - np.fft.ifft(np.abs(np.fft.fft(h_curr))**2).real
        k_max = np.argmax(overlaps)
        
        # Try to reduce overlap at k_max by adjusting h
        # overlap_k = sum_i h[i] * (1-h[i+k])
        # To reduce this, we can increase h[i] where h[i+k] is high, or decrease h[i] where h[i+k] is low
        # But we must maintain sum(h) = N/2 and h in [0,1]
        
        # Simple local move: find i that maximizes h[i] * (1-h[i+k_max]) and try to shift mass
        contribs = h_curr * (1 - np.roll(h_curr, -k_max))
        i_src = np.argmax(contribs)
        # Find i_dst where h is small and h[i_dst + k_max] is large
        # Actually, let's just do a small perturbation
        if h_curr[i_src] > 0.01:
            delta = 0.01
            h_new = h_curr.copy()
            h_new[i_src] -= delta
            # Add delta to some other element
            i_dst = np.argmin(h_curr)
            h_new[i_dst] += delta
            h_new = np.clip(h_new, 0, 1)
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new * (N/2.0) / s
                h_new = np.clip(h_new, 0, 1)
            if objective(h_new) < objective(h_curr):
                h = h_new
    
    return h
# EVOLVE-BLOCK-END
