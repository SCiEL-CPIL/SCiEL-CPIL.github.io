import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 50
DEADLINE = 110.0  # safety margin before 150s kill

def _autoconv(h):
    """Compute discrete autoconvolution of h with itself (full length)."""
    # autoconv[k] = sum_i h[i] * h[i + k] for k in [-(N-1), N-1]
    # Using FFT for efficiency
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(f * np.conj(f), n=size)
    # The autoconvolution centered: conv[i] for i in 0..size-1
    # The zero-lag is at index 0 (or n depending on convention)
    # Actually, ifft(fft(h)*conj(fft(h))) gives the circular autoconvolution
    # We need linear. Let's do it properly.
    # linear autoconv: c[k] = sum_i h[i]*h[i+k]
    # For k>=0: c[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    # For k<0: c[k] = c[-k]
    
    # Using FFT: pad to length >= 2N-1
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(f * np.conj(f), n=size)
    # conv[0] = sum h_i^2 (lag 0)
    # conv[k] for k>0 is the correlation at lag k
    # conv is of length size, but due to circular conv it wraps around
    # The linear autoconvolution is conv[0:N] for non-negative lags and conv[size-N+1:size] for negative lags (conjugate)
    # Actually for real signals, the autoconvolution is symmetric.
    # conv[i] = sum_j h[j] h[j-i] for i in 0..size-1
    # For i=0: sum h[j]^2
    # For i=k>0: sum_{j>=k} h[j] h[j-k]
    # So the non-negative lags are conv[0:N]
    # The negative lags are conv[N:size] which should be the reverse
    return conv

def _score(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * sup_k |autoconv_k|)
    For nonnegative h, autoconv is nonnegative, so sup = max.
    """
    h = np.asarray(h, dtype=np.float64)
    l2sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    conv = _autoconv(h)
    # The supremum of the autoconvolution
    # For nonnegative h, the maximum is at lag 0: sum h_i^2
    # But wait, the problem says "supremum norm of the autoconvolution"
    # If it's the L-infinity norm of the autoconvolution function, and the autoconvolution
    # peaks at lag 0 for nonnegative sequences, then sup = sum h_i^2
    # In that case, ratio = sum h_i^2 / (sum h_i * sum h_i^2) = 1 / sum h_i
    # That would be maximized by minimizing sum h_i, but we have nonnegative heights.
    # If we can scale h, the ratio is scale-invariant? No:
    # ratio = (a^2 sum h_i^2) / ((a sum h_i) * (a^2 sum h_i^2)) = 1/(a sum h_i)
    # So smaller a gives larger ratio. But heights are nonnegative, no upper bound stated.
    # Wait, "Choose 50 nonnegative heights" - if there's no upper bound, we can make them arbitrarily small,
    # ratio -> infinity. That can't be right.
    
    # Re-reading: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # Maybe it's (||h||_2^2) / (||h||_1 * ||autoconv||_infty)
    # For step functions, the autoconvolution is a continuous function. The supremum might not be at lag 0
    # if we consider the continuous version.
    
    # For a step function with heights h_i on intervals [i*d, (i+1)*d], the autoconvolution at shift s is:
    # (f*f)(s) = integral f(t) f(t+s) dt
    # For s in [k*d, (k+1)*d], this is a piecewise linear function.
    # The maximum of the continuous autoconvolution for nonnegative step functions is indeed at s=0,
    # where (f*f)(0) = sum h_i^2 * d (if d=1, sum h_i^2).
    
    # But maybe the problem uses discrete autoconvolution where the supremum is over all lags,
    # and for nonnegative sequences, the max is at lag 0.
    
    # Let me check the seed: h = [1, 0, ..., 0]
    # l2sq = 1, l1 = 1, autoconv at lag 0 = 1
    # ratio = 1 / (1 * 1) = 1. Score = 1.0. This matches.
    
    # If we use h = [1, 1, ..., 1] (all ones)
    # l2sq = 50, l1 = 50
    # autoconv at lag 0 = 50
    # ratio = 50 / (50 * 50) = 0.2
    # That's worse.
    
    # So to maximize, we want l2sq large relative to l1 * max_conv.
    # If max_conv = l2sq (at lag 0), ratio = 1/l1.
    # To maximize 1/l1, we want l1 small. But h nonnegative, if we set all to 0, l1=0, undefined.
    # If we set one to 1 and rest to 0, l1=1, ratio=1.
    # If we set one to 0.5 and rest to 0, l1=0.5, ratio=2? But heights are just numbers, no normalization.
    # Wait, the ratio is scale-invariant in the numerator and denominator?
    # h -> c*h: l2sq -> c^2 l2sq, l1 -> c l1, conv -> c^2 conv
    # ratio -> c^2 l2sq / (c l1 * c^2 conv) = l2sq / (c l1 * conv) = ratio_original / c
    # So smaller c gives larger ratio. We can make c arbitrarily small, ratio -> infinity.
    
    # This suggests there must be a constraint I'm missing, or the "supremum norm of autoconvolution" 
    # is not at lag 0.
    
    # Let me reconsider: maybe the autoconvolution supremum is the L-infinity norm of the 
    # *continuous* autoconvolution, and for step functions, this can be larger than the discrete
    # lag-0 value? No, for nonnegative functions, the autoconvolution is maximized at 0.
    
    # Alternative interpretation: maybe "squared L2 norm" is (||h||_2)^2 = sum h_i^2,
    # "L1 norm" is ||h||_1 = sum h_i, and "supremum norm of autoconvolution" is 
    # max_k |sum_i h_i h_{i+k}| but this is the same as sum h_i^2 for nonnegative.
    
    # I think the problem might be that the autoconvolution supremum is over the *continuous* 
    # shift, and the step function's autoconvolution has a triangular shape at lag 0 that might
    # peak higher? No, at lag 0 it's a rectangle of height sum h_i^2.
    
    # Given the seed scores 1.0 and inspirations score 1.1944, there must be a valid interpretation.
    # Let me assume the ratio is as I defined, and there's an implicit constraint that sum h_i = 1
    # or max h_i = 1, or the heights are in [0,1].
    
    # If max h_i <= 1, then to maximize 1/l1 with max h_i = 1, we set one to 1, rest to 0.
    # ratio = 1. Still 1.
    
    # If sum h_i = 1, then ratio = l2sq / (1 * max_conv).
    # With sum h_i = 1, l2sq <= 1 (when one is 1, rest 0).
    # max_conv >= l2sq (at lag 0).
    # So ratio <= l2sq / l2sq = 1.
    # Still 1.
    
    # I must be misunderstanding the autoconvolution. Let me just compute it as the max over all lags
    # of the discrete autoconvolution, and see if a non-trivial h can beat 1.
    
    max_conv = np.max(np.abs(conv))
    ratio = l2sq / (l1 * max_conv) if l1 > 0 and max_conv > 0 else 0.0
    return ratio

def construct():
    start_time = time.time()
    best_h = np.array([1.0] + [0.0] * (N - 1))
    best_score = _score(best_h)
    
    # Try some structured initializations
    inits = []
    inits.append(np.array([1.0] + [0.0] * (N - 1)))
    inits.append(np.ones(N) / N)
    inits.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]))
    inits.append(np.array([1.0 if i % 3 == 0 else 0.0 for i in range(N)]))
    inits.append(np.exp(-np.linspace(0, 5, N)))
    inits.append(np.exp(-np.linspace(5, 0, N)))
    inits.append(np.sin(np.linspace(0, 4 * np.pi, N)) ** 2)
    inits.append(np.cos(np.linspace(0, 4 * np.pi, N)) ** 2)
    
    # Add random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        h = rng.random(N)
        if rng.random() < 0.5:
            h[h < 0.5] = 0
        inits.append(h)
    
    for h0 in inits:
        if time.time() - start_time > DEADLINE:
            break
        s = _score(h0)
        if s > best_score:
            best_score = s
            best_h = h0.copy()
    
    # Use gradient-based optimization
    def neg_score(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        return -_score(h)
    
    def neg_score_grad(h):
        # Numerical gradient
        eps = 1e-6
        g = np.zeros_like(h)
        for i in range(N):
            hp = h.copy()
            hp[i] += eps
            g[i] = (neg_score(hp) - neg_score(h)) / eps
        return g
    
    # Try a few local optimizations from best initializations
    for h0 in inits[:10]:
        if time.time() - start_time > DEADLINE:
            break
        try:
            res = minimize(neg_score, h0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success and res.fun < -best_score:
                h_opt = np.maximum(res.x, 0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
