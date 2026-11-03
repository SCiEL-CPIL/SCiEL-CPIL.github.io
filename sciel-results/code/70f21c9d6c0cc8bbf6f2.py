import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Use FFT for faster convolution
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    h_fft = np.fft.rfft(h_pad)
    h_conv_fft = h_fft * h_fft
    h_conv = np.fft.irfft(h_conv_fft, n_fft)[:n_conv]
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _find_sidon_supports(k, max_tries=500):
    """Find k-element subsets of [0, N) that are Sidon sets (all pairwise sums distinct)."""
    sidon_sets = []
    for _ in range(max_tries):
        s = set(np.random.choice(N, k, replace=False))
        sums = set()
        is_sidon = True
        for a in s:
            for b in s:
                if a <= b:
                    s_val = a + b
                    if s_val in sums:
                        is_sidon = False
                        break
                    sums.add(s_val)
            if not is_sidon:
                break
        if is_sidon:
            sidon_sets.append(sorted(s))
        if len(sidon_sets) >= 50:
            break
    return sidon_sets

def _soft_max(v, tau=0.1):
    """Soft maximum of vector v."""
    v_max = np.max(v)
    # Prevent overflow
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def _autoconv_component(h, k_idx):
    """Compute (h*h)_k for a specific index k (0 to 2N-2)."""
    lo = max(0, k_idx - N + 1)
    hi = min(N - 1, k_idx)
    if lo > hi:
        return 0.0
    i_vals = np.arange(lo, hi + 1)
    j_vals = k_idx - i_vals
    return np.sum(h[i_vals] * h[j_vals])

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Sidon set initializations - key insight
    for k in range(8, 20):
        if time.time() > deadline: break
        sidon_sets = _find_sidon_supports(k, max_tries=200)
        for support in sidon_sets:
            if time.time() > deadline: break
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0
            # Try different weight patterns
            for weight_type in ['uniform', 'geometric', 'random']:
                if weight_type == 'uniform':
                    h_test = h_sidon.copy()
                elif weight_type == 'geometric':
                    r = np.random.uniform(0.5, 2.0)
                    h_test = h_sidon.copy()
                    for i, idx in enumerate(support):
                        h_test[idx] = r ** i
                else:
                    h_test = h_sidon * np.random.rand(N)
                    h_test[support] = np.random.rand(len(support))
                
                if np.sum(h_test) > 0:
                    h_test /= np.sum(h_test)
                    s = _score_raw(h_test)
                    if s > best_score:
                        best_score = s
                        best_h = h_test.copy()

    # 2. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using L-BFGS-B on log-scale variables with Soft-Max surrogate
    def neg_obj_soft(x):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm = _soft_max(h_conv, tau=0.01)
        if sm == 0:
            return 0.0
        return -(l2_sq / sm)

    h_start = best_h.copy()
    h_start[h_start < 1e-8] = 1e-8
    x0 = np.log(h_start)
    
    try:
        res = minimize(neg_obj_soft, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success:
            h_candidate = np.exp(res.x)
            h_candidate /= np.sum(h_candidate)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass

    # Try multiple restarts
    inits = [best_h.copy()]
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    for h_init in inits:
        if time.time() > deadline: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        h_init[h_init < 1e-8] = 1e-8
        x_init = np.log(h_init)
        
        try:
            res = minimize(neg_obj_soft, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # NEW: Iterative peak-index refinement (fixed-point on argmax)
    if time.time() < deadline - 30:
        h_ref = best_h / np.sum(best_h)
        
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            
            # Use FFT for speed
            n_conv = 2 * N - 1
            n_fft = 1
            while n_fft < n_conv:
                n_fft *= 2
            h_pad = np.zeros(n_fft)
            h_pad[:N] = h_ref
            h_fft = np.fft.rfft(h_pad)
            h_conv = np.fft.irfft(h_fft * h_fft, n_fft)[:n_conv]
            peak_idx = np.argmax(h_conv)
            
            def neg_obj_fixed_peak(x, k_idx):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                val = _autoconv_component(h, k_idx)
                if val <= 0:
                    return 0.0
                return -(l2_sq / val)
            
            h_ref[h_ref < 1e-8] = 1e-8
            x_ref = np.log(h_ref)
            
            try:
                res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                              method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                    h_ref = h_candidate
            except Exception:
                break

    # Final polish with tighter tolerance on the exact (non-soft) objective
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_exact(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            # Use FFT for speed
            n_conv = 2 * N - 1
            n_fft = 1
            while n_fft < n_conv:
                n_fft *= 2
            h_pad = np.zeros(n_fft)
            h_pad[:N] = h
            h_fft = np.fft.rfft(h_pad)
            h_conv = np.fft.irfft(h_fft * h_fft, n_fft)[:n_conv]
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / max_conv)

        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-15})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Local search: perturb individual heights
    if time.time() < deadline - 3:
        h_local = best_h.copy()
        for _ in range(500):
            if time.time() > deadline - 1:
                break
            i = np.random.randint(N)
            old_val = h_local[i]
            for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                h_local[i] = old_val * factor
                if np.sum(h_local) == 0:
                    continue
                h_local /= np.sum(h_local)
                s = _score_raw(h_local)
                if s > best_score:
                    best_score = s
                    best_h = h_local.copy()
                    break
            else:
                h_local[i] = old_val

    return list(best_h)

# EVOLVE-BLOCK-END
