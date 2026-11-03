"""Combined: Incremental local search + smooth surrogate optimization for signed step heights."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp. Returns (value, weights)."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=50.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    Uses FFT-based cross-correlation for the autoconvolution gradient.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)

    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)

    if weights is None:
        return np.zeros_like(h)

    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)

    # Gradient of S = soft_max_abs(ac) w.r.t. h:
    # dS/dh[j] = sum_m g_ac[m] * (h[m-j] + h[j-m])
    # By symmetry, this equals 2 * cross_corr(g_ac, h)[j]
    # where cross_corr(a, b)[j] = sum_m a[m+j] * b[m]
    
    # Compute cross-correlation via FFT
    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2

    V = np.fft.rfft(g_ac, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)
    
    # C[j] = sum_n g_ac[n+j] * h[n]
    grad_S = 2.0 * C[:N]

    # f = S / s^2
    # df/dh[j] = (grad_S[j] * s - 2 * S) / s^3
    df = (grad_S * s - 2.0 * sm) / (s ** 3)

    return df

def _lbfgs_optimize(h, beta=50.0, max_iter=300, deadline=None):
    """L-BFGS optimization of the smooth surrogate."""
    h0 = h.copy()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)
    
    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 1000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)

def _single_flip_local_search_incremental(h, max_passes=20, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    Uses incremental autoconvolution update for O(N) per flip instead of O(N log N).
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv(h)
    best_val = float(np.max(np.abs(ac))) / (s * s)
    
    # Precompute indices for incremental updates
    # ac[k] = sum_j h[j] h[k-j]
    # Flip h[i] -> -h[i]:
    # delta_ac[k] = -2 h[i] (h[k-i] + h[i-k])
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            hi = h[i]
            s_new = s - 2.0 * hi
            
            # Calculate delta_ac
            # We need to update ac in place or compute new max efficiently
            # delta_ac is sparse-ish, but effectively touches 2N elements.
            # Instead of creating a new array, we can compute the new max by checking affected indices.
            # However, for simplicity and robustness within the time limit, we can update a copy 
            # or just compute the new max on the fly. 
            # Given N=400, creating a temporary array of size 799 is cheap.
            
            ac_new = ac.copy()
            
            # term1: k in [i, i+N-1], h[k-i]
            k1_start = i
            k1_end = min(i + N, 2 * N - 1)
            if k1_start < k1_end:
                # h[k-i] for k in k1_start..k1_end-1 is h[0..k1_end-k1_start-1]
                ac_new[k1_start:k1_end] -= 2.0 * hi * h[0:k1_end - k1_start]
            
            # term2: k in [max(0,i-N+1), i], h[i-k]
            k2_start = max(0, i - N + 1)
            k2_end = i + 1
            if k2_start < k2_end:
                # h[i-k] for k in k2_start..k2_end-1
                # indices: i-k2_start ... i-(k2_end-1)
                idx = i - np.arange(k2_start, k2_end)
                ac_new[k2_start:k2_end] -= 2.0 * hi * h[idx]
            
            new_val = float(np.max(np.abs(ac_new))) / (s_new * s_new)
            
            if new_val < best_val:
                h[i] = -hi
                s = s_new
                ac = ac_new
                best_val = new_val
                improved = True
        
        if not improved:
            break
    
    return h, best_val

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_window(n, window_type='gaussian'):
    x = np.arange(n)
    if window_type == 'gaussian':
        mu = n / 2
        sigma = n / 4
        return np.exp(-(x - mu)**2 / (2 * sigma**2))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n - 1))
    elif window_type == 'triangle':
        return 1.0 - np.abs(x - n/2) / (n/2)
    return np.ones(n)

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)

    # Baseline: all ones
    base_score = _raw_value(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate diverse binary candidates
    candidates = []
    t = np.arange(N)
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Chirp sequences (quadratic phase)
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1.0
        candidates.append(x_chirp)
    
    # 3. Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1.0
        candidates.append(x_lin)
    
    # 4. Barker-like structured
    x_barker = np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)])
    candidates.append(x_barker)
    
    # 5. Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1.0
    candidates.append(x_gold)
    
    # 6. Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # 7. Smoothed random (continuous, then binarize)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        noise = rng.standard_normal(N)
        kernel = np.ones(10) / 10
        smoothed = np.convolve(noise, kernel, mode='same')
        x_smooth = np.sign(smoothed)
        x_smooth[x_smooth == 0] = 1.0
        candidates.append(x_smooth)
        
    # 8. Gaussian-windowed random
    for seed in [11, 22, 33]:
        noise = rng.standard_normal(N)
        window = _gen_window(N, 'gaussian')
        x_gauss = np.sign(noise * window)
        x_gauss[x_gauss == 0] = 1.0
        candidates.append(x_gauss)

    # Evaluate all candidates
    scored_candidates = []
    for x in candidates:
        val = _raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Update best if any candidate is better
    if scored_candidates[0][0] < best_score:
        best_score = scored_candidates[0][0]
        best_h = scored_candidates[0][1].copy()

    # Run incremental local search on top candidates
    # We can afford more passes now
    top_k = 10
    for init_val, h_init in scored_candidates[:top_k]:
        if time.time() > deadline - 20:
            break
        
        # If initial value is already worse than best, we might still improve it,
        # but if it's significantly worse, we might skip. 
        # However, local search can jump out of bad local minima if the landscape is rugged.
        # Let's try the top 5 most promising ones.
        
        h_work, val_work = _single_flip_local_search_incremental(h_init, max_passes=20, deadline=deadline - 10)
        if val_work < best_score:
            best_score = val_work
            best_h = h_work.copy()
        
        # Check time
        if time.time() > deadline - 15:
            break

    # Hybrid: Use best binary result to seed L-BFGS
    if time.time() < deadline - 30:
        h_seed = best_h.copy()
        s0 = np.sum(h_seed)
        if abs(s0) > 1e-6:
            h_norm = h_seed / s0
            
            # Try L-BFGS with different betas
            for beta in [30, 60]:
                if time.time() > deadline - 15:
                    break
                h_lbfgs, _ = _lbfgs_optimize(h_norm, beta=beta, max_iter=100, deadline=deadline - 5)
                s_lbfgs = np.sum(h_lbfgs)
                if abs(s_lbfgs) > 1e-6:
                    h_lbfgs = h_lbfgs / s_lbfgs
                    val_lbfgs = _raw_value(h_lbfgs)
                    if val_lbfgs < best_score:
                        best_score = val_lbfgs
                        best_h = h_lbfgs.copy()
                    
                    # Binarize and refine
                    h_bin = np.sign(h_lbfgs)
                    h_bin[h_bin == 0] = 1.0
                    val_bin = _raw_value(h_bin)
                    if val_bin < best_score:
                        best_score = val_bin
                        best_h = h_bin.copy()
                        
                    # Final polish on the binarized version
                    if time.time() < deadline - 5:
                        h_work, val_work = _single_flip_local_search_incremental(h_bin, max_passes=10, deadline=deadline - 2)
                        if val_work < best_score:
                            best_score = val_work
                            best_h = h_work.copy()

    # Final polish on the overall best result
    if time.time() < deadline - 2:
        h_final = np.sign(best_h)
        h_final[h_final == 0] = 1.0
        h_final, val_final = _single_flip_local_search_incremental(h_final, max_passes=5, deadline=deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
