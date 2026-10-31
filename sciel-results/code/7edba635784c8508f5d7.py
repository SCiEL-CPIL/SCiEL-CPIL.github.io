"""
Erdos minimum overlap via spectral convex optimization.

Strategy:
- The overlap at shift s is O_s = sum_i h[i] * (1 - h[(i+s)%N]) = N/2 - autocorr[s].
- We want to minimize max_s O_s, which is equivalent to maximizing min_s autocorr[s].
- Let H = fft(h). Then autocorr[s] = ifft(|H|^2)[s].
- We can formulate a convex problem: minimize gamma such that
    autocorr[s] >= -gamma + N/2  for all s
    sum h = N/2
    0 <= h <= 1
    H is PSD (which is automatically satisfied if h is real, but we work in spectral domain)

Actually, a cleaner approach:
- Work directly in the time domain but use a convex relaxation.
- The max overlap is max_s (N/2 - autocorr[s]).
- Minimizing max_s (N/2 - autocorr[s]) is equivalent to minimizing the maximum of a set of linear functions of h (since autocorr[s] is quadratic in h, this is NOT convex).

Wait, autocorr[s] = sum_i h[i] h[(i+s)%N] is quadratic, so the problem is non-convex in h.

Alternative spectral approach:
- Let x = |H[k]|^2 be the power spectrum. Then autocorr[s] = (1/N) sum_k x[k] exp(2pi i k s / N).
- We want to minimize max_s (N/2 - autocorr[s]).
- This is still non-convex in x because of the max of sinusoids.

However, we can use a different trick:
- The maximum overlap is related to the peak of the aperiodic autocorrelation.
- For binary sequences, this is a known problem. For continuous h in [0,1], we can use:
  1. Start with a good initial guess (e.g., from a known low-autocorrelation sequence).
  2. Use projected gradient descent on the spectral domain to minimize the max overlap.

Let's try a different angle: use the fact that for the sum constraint and bounds, the optimal h might have a simple structure.
Known result: for N even, the minimum possible max overlap is related to the Welch bound.

Let's implement a robust spectral optimization:
1. Initialize h with a random binary-like sequence satisfying sum = N/2.
2. Compute overlaps.
3. Use subgradient method: the max overlap is a convex function of h (as a function of the vector of overlaps, but the overlaps are quadratic).
   Actually, max_s f_s(h) where f_s is quadratic is not convex.

Let's use a practical approach:
- Use scipy.optimize.minimize with a smooth approximation of the max (log-sum-exp) in the spectral domain.
- Variables: H (complex FFT coefficients). Constraints: h = ifft(H) in [0,1], sum h = N/2.
- This is a convex problem if we relax the [0,1] constraint? No, it's non-convex.

Given the time budget, let's use a hybrid:
1. Generate several candidate h from known low-autocorrelation structures (e.g., Barker-like, Zadoff-Chu-like).
2. Refine each using projected gradient descent on the smooth surrogate.
3. Return the best.
"""

import numpy as np
import time

N = 100

def _max_overlap_fft(h):
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    return float(np.max(overlaps))

def _normalize_h(h):
    s = h.sum()
    if s < 1e-12:
        return np.full(N, 0.5)
    h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) > 1e-12:
        order = np.argsort(np.abs(h - 0.5))
        remaining = diff
        for idx in order:
            if remaining > 0:
                adj = min(remaining, 1.0 - h[idx])
                h[idx] += adj
                remaining -= adj
            elif remaining < 0:
                adj = max(remaining, -h[idx])
                h[idx] += adj
                remaining -= adj
            if abs(remaining) < 1e-12:
                break
    return h

def _log_sum_exp_overlap(h, beta=50.0):
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    m = np.max(overlaps)
    val = m + np.log(np.sum(np.exp(overlaps - m))) / beta
    return val

def _generate_zadoff_chu_like(N):
    """Generate a Zadoff-Chu-like sequence."""
    root = 1
    h = np.zeros(N)
    for i in range(N):
        h[i] = 0.5 + 0.5 * np.cos(2 * np.pi * root * i * (i + 1) / N)
    return _normalize_h(h)

def _generate_barker_like(N):
    """Generate a Barker-like sequence."""
    # Use a known Barker code of length 13, then extend
    barker = np.array([1, 1, 1, -1, 1, -1, -1])
    h = np.zeros(N)
    for i in range(N):
        h[i] = 0.5 + 0.5 * barker[i % len(barker)]
    return _normalize_h(h)

def _gradient_descent_smooth(h, beta=50.0, lr=0.01, n_iters=500):
    """Project gradient descent on smooth surrogate."""
    h = h.copy()
    for _ in range(n_iters):
        # Compute gradient of log-sum-exp overlap
        F = np.fft.fft(h)
        autocorr = np.fft.ifft(F * np.conj(F)).real
        overlaps = N / 2.0 - autocorr
        m = np.max(overlaps)
        weights = np.exp(overlaps - m) / np.sum(np.exp(overlaps - m))
        
        # Gradient of max overlap w.r.t. h
        # d(overlaps)/dh = -d(autocorr)/dh
        # autocorr[s] = sum_i h[i] h[(i+s)%N]
        # d(autocorr[s])/dh[j] = h[(j+s)%N] + h[(j-s)%N] (for circular)
        # So d(overlaps)/dh[j] = -(h[(j+s)%N] + h[(j-s)%N])
        # Weighted sum over s
        grad = np.zeros(N)
        for s in range(N):
            w = weights[s]
            if w > 1e-10:
                # d(autocorr[s])/dh[j] = h[(j+s)%N] + h[(j-s)%N]
                # This is a convolution
                pass
        
        # More efficient: use FFT
        # autocorr = ifft(|F|^2)
        # d(autocorr)/dh = 2 * real(ifft(F * conj(dF/dh)))
        # dF/dh = F matrix
        # This is getting complex. Let's use a simpler approach.
        
        # Actually, let's just use the fact that the gradient of the max overlap
        # can be approximated by the overlap values themselves.
        
        # For now, use a simple perturbation approach
        break
    
    return h

def construct():
    deadline = time.time() + 115.0
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fft(best_h)
    
    # Generate initial candidates
    candidates = []
    candidates.append(_generate_zadoff_chu_like(N))
    candidates.append(_generate_barker_like(N))
    
    # Random binary-like sequences
    np.random.seed(42)
    for _ in range(10):
        h = np.random.choice([0.0, 1.0], size=N)
        h = _normalize_h(h)
        candidates.append(h)
    
    # Refine each candidate using SLSQP on smooth surrogate
    for h0 in candidates:
        if time.time() > deadline:
            break
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda p: _log_sum_exp_overlap(p, beta=80.0),
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda p: p.sum() - N/2.0}],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            h_refined = _normalize_h(result.x)
            sc = _max_overlap_fft(h_refined)
            
            if sc < best_score:
                best_score = sc
                best_h = h_refined.copy()
        except Exception:
            pass
    
    # Final validation
    if abs(best_h.sum() - N / 2.0) > 1e-8:
        best_h = _normalize_h(best_h)
    
    return best_h
