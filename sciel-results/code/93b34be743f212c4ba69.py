"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation with a smooth-max (log-sum-exp) surrogate.
Multi-phase optimization: coarse-to-fine beta schedule, multiple starts, and
analytical structure exploitation.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
# Evaluator kills at 150s. Target finishing by 142s.
_DEADLINE = time.time() + 142


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _overlap_grad(h, w):
    """
    Compute sum_k w[k] * dC_k/dh using FFT-based convolution.
    C_k = 1 - (2/n) * r_k,  r_k = sum_{i>=k} h_i * h_{i-k}
    dC_k/dh_j = -(2/n) * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
    Vectorized: grad_j = -(2/n) * (sum_{k<=j} w_k h_{j-k} + sum_{k<n-j} w_k h_{j+k})
    The first term is a linear convolution (w * h), the second is a cross-correlation.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    H = np.fft.rfft(h_pad, n=n_fft)
    W = np.fft.rfft(w_pad, n=n_fft)
    # Linear convolution w * h: (w*h)_j = sum_k w_k h_{j-k} for j>=k
    conv = np.fft.irfft(W * H, n=n_fft)[:n]
    # Cross-correlation: sum_k w_k h_{j+k} = (w * h_rev)_j where h_rev[i] = h[n-1-i]
    h_rev = h[::-1]
    h_rev_pad = np.zeros(n_fft)
    h_rev_pad[:n] = h_rev
    H_rev = np.fft.rfft(h_rev_pad, n=n_fft)
    corr = np.fft.irfft(W * H_rev, n=n_fft)[:n]
    # corr[j] = sum_k w_k h_rev[j-k] = sum_k w_k h[n-1-(j-k)] = sum_k w_k h[n-1-j+k]
    # We want sum_k w_k h[j+k]. Let m = j+k, k = m-j. sum_m w_{m-j} h[m].
    # corr[j] = sum_k w_k h[n-1-j+k]. Let l = n-1-j+k => k = l - (n-1-j) = l + j - n + 1
    # corr[j] = sum_l w_{l+j-n+1} h[l]
    # We need sum_k w_k h[j+k] = sum_l w_{l-j} h[l] where l = j+k
    # So we need (w * h)_l type but with w indexed by l-j, i.e. correlation of w with h.
    # Actually let's just use: sum_k w_k h[j+k] for k=0..n-1-j
    # This equals the cross-correlation of w and h at lag j.
    # Cross-correlation: (w ⋆ h)[j] = sum_k w[k] h[k+j]
    # In FFT: ifft(fft(w) * conj(fft(h)))... let me just compute directly via the other way.
    # corr2[j] = sum_k w[k] h[k+j]. FFT: ifft(conj(W) * H) gives sum_k w[k] h[k+j]? 
    # Actually: ifft(conj(W)*H)[j] = sum_k w[k] h[(k-j) mod N]... not quite.
    # Let me use: ifft(W * conj(H))[j] = sum_k w[k] conj(h)[(j-k) mod N] = sum_k w[k] h[j-k]
    # That's the same as conv. 
    # For sum_k w[k] h[k+j]: use ifft(conj(W) * H)[j] = sum_k conj(w)[k] h[(j-k) mod N]
    # Hmm, let me just compute it as: reverse w, convolve, reverse back.
    # sum_k w[k] h[k+j] = sum_m w[m-j] h[m] where m=k+j, so m from j to n-1
    # = (w_rev * h)[n-1-j] where w_rev[i] = w[n-1-i]
    # Let's verify: (w_rev * h)[p] = sum_i w_rev[i] h[p-i] = sum_i w[n-1-i] h[p-i]
    # Let m = n-1-i, i = n-1-m: sum_m w[m] h[p-n+1+m]
    # Set p = n-1-j: sum_m w[m] h[m-j]. That's sum_m w[m] h[m-j], not what we want.
    # We want sum_k w[k] h[k+j] = sum_m w[m] h[m+j] (renaming k->m)
    # (w_rev * h)[p] = sum_m w[m] h[p-n+1+m]. Set p-n+1 = j => p = n-1+j.
    # So (w_rev * h)[n-1+j] = sum_m w[m] h[m+j]. 
    # But p = n-1+j can be >= n, so we need the full convolution.
    # This is getting complicated. Let me just use a simpler O(N^2) approach but vectorized with numpy.
    # Actually for N=100, even a vectorized O(N^2) is fine. Let me use matrix approach.
    
    # Simple vectorized approach:
    # grad_j = -(2/n) * (sum_{k=0}^{j} w_k h_{j-k} + sum_{k=0}^{n-1-j} w_k h_{j+k})
    # First sum: conv of w and h, first n elements
    # Second sum: for each j, sum_{k=0}^{n-1-j} w_k h_{j+k}
    # = sum_{k=0}^{n-1} w_k h_{j+k} * (j+k < n)
    # Build matrix M where M[j,k] = h[j+k] if j+k < n else 0
    # Then second_sum = M @ w
    
    # Actually, let me just do it the straightforward vectorized way for N=100:
    # Use the FFT approach properly.
    # sum_k w[k] h[k+j] for k=0..n-1-j
    # = cross-correlation of w and h at shift j
    # In FFT: ifft(fft(w) * conj(fft(h))) gives circular cross-correlation
    # (w ⋆ h)[j] = sum_k w[k] h[(k+j) mod n]
    # For our case with zero-padding to n_fft, this gives linear cross-correlation.
    
    W_full = np.fft.rfft(w_pad, n=n_fft)
    H_full = np.fft.rfft(h_pad, n=n_fft)
    # Cross-correlation: ifft(conj(W) * H)
    xc = np.fft.irfft(np.conj(W_full) * H_full, n=n_fft)[:n]
    # xc[j] = sum_k w[k] h[(k+j) mod n_fft] but with zero padding it's sum_k w[k] h[k+j] for k+j < n
    # Wait: ifft(conj(W)*H)[j] = sum_k w[k] * h[j-k]... no.
    # ifft(conj(W)*H)[j] = (1/n_fft) sum_m conj(W[m]) H[m] exp(2πimj/n_fft)
    # = sum_k w[k] h[(j-k) mod n_fft]  ... this is correlation not what we want.
    
    # Let me just use the direct numpy approach. For N=100 it's fast enough.
    idx = np.arange(n)
    # First term: (w * h)_j = sum_{k=0}^{j} w_k h_{j-k}
    # This is just np.convolve(w, h, 'full')[:n]
    term1 = np.convolve(w, h, 'full')[:n]
    # Second term: sum_{k=0}^{n-1-j} w_k h_{j+k}
    # Build upper-triangular matrix
    # S[j] = sum_{k=0}^{n-1-j} w[k] h[j+k]
    # = sum_{k} w[k] h[j+k] * (j+k < n)
    # Using broadcasting:
    # For each j, we need dot(w[:n-j], h[j:])
    # This is like a reversed convolution.
    # h[j:] dotted with w[:n-j]
    # Let's build it: create matrix where row j is [w[0], w[1], ..., w[n-1-j], 0, ..., 0]
    # and columns are h[0], h[1], ..., h[n-1]
    # Actually simpler: use np.correlate or manual
    # S[j] = sum_k w[k] h[j+k] = (w ⋆ h)[j] (cross-correlation)
    # np.correlate(h, w, 'full') gives sum_k h[k] w[k-j]... let me check.
    # np.correlate(a, v, mode) computes sum_k a[k+m] conj(v[k])
    # So np.correlate(h, w, 'full')[j] = sum_k h[k + (j - n + 1)] w[k]... offset is tricky.
    # Let me just do it with a loop over j but vectorized inner product. For N=100, 100 dot products of length up to 100 is trivial.
    term2 = np.zeros(n)
    for j in range(n):
        term2[j] = np.dot(w[:n - j], h[j:])
    
    grad = -(2.0 / n) * (term1 + term2)
    return grad


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    
    return h


def _run_slsqp_epigraph(h0, maxiter, ftol):
    """
    Run SLSQP with epigraph formulation:
    minimize t
    s.t. C_k(h) <= t  for all k
         sum(h) = N/2
         0 <= h_i <= 1
    Variable vector: [h_0, ..., h_{N-1}, t]
    """
    n = N
    # Variable: x = [h (n), t (1)]
    x0 = np.concatenate([h0, [float(_score(h0)) + 0.1]])
    
    bounds = [(0.0, 1.0)] * n + [(-1.0, 3.0)]  # t can be negative in principle
    
    def _cs(x):
        return _linear_overlap_values(x[:n])
    
    def obj_func(x):
        return x[n]  # t
    
    def jac_func(x):
        g = np.zeros(n + 1)
        g[n] = 1.0
        return g
    
    # Equality constraint: sum(h) = N/2
    def eq_fun(x):
        return np.sum(x[:n]) - N / 2.0
    
    def eq_jac(x):
        j = np.zeros(n + 1)
        j[:n] = 1.0
        return j
    
    # Inequality constraints: C_k(h) - t <= 0
    def ineq_fun(x):
        cs = _cs(x[:n])
        return cs - x[n]
    
    def ineq_jac(x):
        h = x[:n]
        # d(C_k - t)/dx = [dC_k/dh, -1]
        # We need gradient of each C_k with respect to h
        # dC_k/dh_j = -(2/n) * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
        # Build Jacobian: shape (n, n+1)
        J = np.zeros((n, n + 1))
        J[:, n] = -1.0
        # For each k, dC_k/dh:
        # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
        for k in range(n):
            factor = -(2.0 / n)
            if k > 0:
                # j >= k: dC_k/dh_j includes factor * h_{j-k}
                # For j from k to n-1: h_{j-k} = h[0], h[1], ..., h[n-1-k]
                J[k, k:] += factor * h[:n - k]
            # j from 0 to n-1-k: dC_k/dh_j includes factor * h_{j+k}
            # h[j+k] for j=0..n-1-k = h[k], h[k+1], ..., h[n-1]
            J[k, :n - k] += factor * h[k:]
        return J
    
    cons = [{'type': 'eq', 'fun': eq_fun, 'jac': eq_jac},
            {'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_jac}]
    
    try:
        res = minimize(obj_func, x0, jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.x is not None:
            return res.x[:n]
        return h0
    except Exception:
        return h0


def _try_analytical_form():
    """
    Try structured forms that might be near-optimal.
    Based on the observation that for the constant h=0.5, all C_k = 1.
    We want to reduce the max C_k below 1.
    
    The key insight: C_k = 1 - (2/N) * r_k, so minimizing max C_k 
    is equivalent to maximizing min r_k.
    
    Try cosine-like profiles with different frequencies.
    """
    candidates = []
    i = np.arange(N)
    
    # Cosine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5, 0.55]:
        h = 0.5 + amp * np.cos(2.0 * np.pi * i / N)
        candidates.append(h)
    
    # Sine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5]:
        h = 0.5 + amp * np.sin(2.0 * np.pi * i / N)
        candidates.append(h)
    
    # Second harmonic
    for amp in [0.2, 0.3, 0.4]:
        h = 0.5 + amp * np.cos(4.0 * np.pi * i / N)
        candidates.append(h)
    
    # Mix of harmonics
    h = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)
    
    h = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) - 0.1 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)
    
    # Square-wave-like
    h = np.where(i % 2 == 0, 0.7, 0.3)
    candidates.append(h)
    
    h = np.where(i % 2 == 0, 0.6, 0.4)
    candidates.append(h)
    
    # Triangular wave
    phase = 2.0 * np.pi * i / N
    h = 0.5 + (2.0 / np.pi) * np.arcsin(np.sin(phase)) * 0.4
    candidates.append(h)
    
    return candidates


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Generate initial candidates
    i = np.arange(N)
    candidates = []
    
    # Constant
    candidates.append(np.full(N, 0.5))
    
    # Analytical structured forms
    candidates.extend(_try_analytical_form())
    
    # Random starts
    for _ in range(8):
        h = np.random.uniform(0.05, 0.95, N)
        candidates.append(h)
    
    # Evaluate all candidates
    for h0 in candidates:
        _update_best(h0)
    
    # 2. Epigraph-based optimization (smoother than log-sum-exp for SLSQP)
    # Phase 1: Broad exploration with moderate iterations
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _run_slsqp_epigraph(best_h, maxiter=300, ftol=1e-6)
        _update_best(h_opt)
    
    # Phase 2: Refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _run_slsqp_epigraph(best_h, maxiter=500, ftol=1e-8)
        _update_best(h_opt)
    
    # Phase 3: Further refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 12:
        h_opt = _run_slsqp_epigraph(best_h, maxiter=700, ftol=1e-9)
        _update_best(h_opt)
    
    # Phase 4: High-precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 6:
        h_opt = _run_slsqp_epigraph(best_h, maxiter=1000, ftol=1e-10)
        _update_best(h_opt)
    
    # Phase 5: Final emergency polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _run_slsqp_epigraph(best_h, maxiter=500, ftol=1e-11)
        _update_best(h_opt)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
