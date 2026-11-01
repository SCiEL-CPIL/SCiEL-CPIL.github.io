"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_acf(h):
    """Compute linear aperiodic autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft).real[:N]
    return ac

def _smooth_max_sidelobe(x, beta=50.0):
    """
    Smooth approximation of max |acf[k]| for k in 1..N-1.
    x is a continuous vector in [-1, 1].
    """
    ac = _compute_acf(x)
    sidelobes = np.abs(ac[1:])
    if len(sidelobes) == 0:
        return 0.0, np.zeros_like(x)
    
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return 0.0, np.zeros_like(x)
        
    # Log-sum-exp for stability
    shifted = beta * (sidelobes - max_val)
    # Clip to prevent overflow, though shifted <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    val = max_val + log_sum / beta
    
    # Gradient calculation
    # d/dx_i soft_max = sum_k w_k * sign(ac[k]) * (x[i+k] + x[i-k])
    # where w_k = exp(beta*(ac[k]-max)) / sum_exp = beta * exp(shifted[k]) / sum_exp
    weights = beta * np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros(N)
    # We only need to compute gradient for significant weights to save time
    # But for L-BFGS, we need the full gradient.
    # Vectorized approach for gradient of max |acf| is tricky, 
    # but we can approximate by taking the top K lags.
    # For beta=50, the top few lags dominate.
    top_k = 10
    idx = np.argsort(weights)[::-1][:top_k]
    
    for k_idx in idx:
        k = k_idx + 1
        w_k = weights[k_idx]
        if w_k < 1e-6:
            continue
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        if c_k == 0:
            continue
        grad[0:N-k] += c_k * x[k:N]
        grad[k:N] += c_k * x[0:N-k]
        
    return val, grad

def _local_search_psl(h, deadline):
    """Greedy local search to minimize Peak Sidelobe Level."""
    best_h = h.copy()
    ac = _compute_acf(best_h)
    best_val = np.max(np.abs(ac[1:]))
    
    # Precompute indices for faster flip evaluation? 
    # Flipping bit i changes acf[k] by -2*h[i]*h[i+k] - 2*h[i]*h[i-k] (with sign changes)
    # Specifically, if h[i] flips, acf[k] changes by -2 * h[i] * (h[i+k] + h[i-k]) (if valid)
    # Actually, acf_new[k] = acf_old[k] - 2*h[i]*(h[i+k] + h[i-k]) (terms involving i)
    # This is O(N) per flip if we update the whole acf, or O(N) to check if max changes.
    # Simpler: just recompute acf via FFT for each candidate flip? 
    # N=400, FFT is fast. 400 flips * 400 iterations * FFT cost is manageable.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            # Try flip
            old_val = best_h[i]
            best_h[i] = -old_val
            
            ac = _compute_acf(best_h)
            val = np.max(np.abs(ac[1:]))
            
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = old_val # revert
                
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_h = np.ones(N)
    ac_best = _compute_acf(best_h)
    best_val = np.max(np.abs(ac_best[1:]))
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous Optimization
    # Try a few different initializations
    inits = [
        np.ones(N),
        rng.choice([-1.0, 1.0], size=N),
        np.sin(np.linspace(0, 4*np.pi, N)), # Structured
    ]
    
    for x0 in inits:
        if time.time() > deadline - 30:
            break
            
        # Ensure x0 is in [-1, 1]
        x0 = np.clip(x0, -1.0, 1.0)
        
        try:
            res = minimize(
                _smooth_max_sidelobe, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            x_opt = res.x
            # Round to binary
            h_bin = np.sign(x_opt)
            h_bin[h_bin == 0] = 1.0
            
            ac = _compute_acf(h_bin)
            val = np.max(np.abs(ac[1:]))
            
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
            # Quick local search refinement
            if time.time() < deadline - 10:
                h_ref, val_ref = _local_search_psl(h_bin.copy(), deadline - 10)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
        except Exception:
            continue
            
    # Phase 2: Final Polish
    if time.time() < deadline - 5:
        h_final, val_final = _local_search_psl(best_h.copy(), deadline - 5)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    return best_h
# EVOLVE-BLOCK-END
