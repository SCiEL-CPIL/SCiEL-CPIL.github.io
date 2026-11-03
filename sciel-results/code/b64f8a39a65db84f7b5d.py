"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Combined approach:
- Compact basis parameterization + global optimization (DE, L-BFGS-B, basin hopping)
- O(N) fast incremental autoconvolution update for local search (from inspiration)
- Smooth log-sum-exp surrogate with adaptive beta for gradient-based refinement
- Peak-suppression phase targeting the argmax of the autoconvolution
- Sign-flip local search with efficient incremental updates
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.signal import fftconvolve

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _soft_max_abs(ac, beta=50.0):
    """
    Soft approximation of max|ac| using log-sum-exp for differentiability.
    Returns a smooth surrogate that approximates max|ac|.
    """
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _score_smooth(h, beta=50.0):
    """
    Smooth surrogate for the objective: soft_max|autoconv| / (sum h)^2.
    This is differentiable and can be used with gradient-based optimizers.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _make_basis_functions(n, n_basis=25):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 8):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 8):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions
    for center_frac in [0.2, 0.4, 0.6, 0.8]:
        mu = center_frac * n
        sigma = n / 8
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosine
    for k in [10, 15, 20]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    return np.array(basis[:n_basis])

def _flip_score_fast(h, idx, ac, cur_sum):
    """
    Compute the new max|autoconv|/sum^2 after flipping h[idx] in O(N) time.
    The autoconvolution change for flipping h[idx] from v to -v is:
      delta_ac[k] = -4*v*h[k-idx] for k in [idx, idx+N-1]
      delta_ac[idx] += 4*v^2  (correction for the double-counted term)
    We track the current max and update it incrementally.
    """
    v = h[idx]
    new_sum = cur_sum - 2.0 * v
    nsq = new_sum * new_sum
    if nsq < 1e-10:
        return np.inf
    
    # The change to ac at lag k (for k from idx to idx+N-1) is:
    # new_ac[k] = ac[k] - 4*v*h[k-idx]
    # But at k=idx, we also add 4*v^2 because the original ac[idx] had v*h[idx] 
    # contributing, and now it's (-v)*(-v) = v^2 vs v*v = v^2... 
    # Actually let's re-derive:
    # ac[k] = sum_{i} h[i]*h[i+k]
    # After flip: h[idx] -> -h[idx]
    # new_ac[k] = ac[k] - 2*v*h[idx+k] - 2*v*h[idx+k] + ... 
    # Wait, the flip changes h[idx] from v to -v, delta = -2v
    # new_ac[k] = sum h'[i]*h'[i+k] where h' differs only at idx
    # = ac[k] - 2v*(h[idx+k] if idx+k != idx else 0) - 2v*(h[idx] if idx != idx+k else 0) + ...
    # More carefully:
    # new_ac[k] = ac[k] - 2v*h[idx+k] - 2v*h[k-idx] + 4v^2*[k==idx]
    # where we interpret h[j]=0 if j out of [0,N-1]
    
    # For k in [0, N-1]:
    #   h[idx+k] is valid if idx+k < N, i.e., k < N-idx
    #   h[k-idx] is valid if k-idx >= 0, i.e., k >= idx
    # So for k < idx: only h[k-idx] term (but k-idx < 0, so 0) -> change = 0... 
    # Wait no. Let me be more careful.
    
    # ac[k] = sum_{i=max(0,k-N+1)}^{min(N-1, N-1-k)} h[i]*h[i+k]  for k>=0
    # Actually for linear autoconv: ac[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k] for k=0..N-1
    # and ac[-k] = ac[k] by symmetry (but we compute full 2N-1)
    
    # For the full (2N-1) autoconv with index shift:
    # ac_full[m] = sum_{i} h[i]*h[i+m-(N-1)] ... this is getting complex.
    
    # Let me just use the FFT-based incremental approach but avoid the copy:
    # Compute the delta vector and find max of |ac + delta|
    
    # delta_ac for lag m (0 to 2N-2):
    # The contribution of h[idx] to ac_full[m] is h[idx]*h[idx + m - (N-1)] 
    # Wait, this is the standard linear convolution form.
    
    # Let me just compute it directly for the relevant range.
    # h[idx] appears in ac_full[m] when there exists i such that i = idx and i + (m - (N-1)) is in [0,N-1]
    # OR i + (m-(N-1)) = idx and i is in [0,N-1]
    # So m - (N-1) = j - idx for some valid j, i.e., m = N-1 + j - idx
    # j in [0,N-1] means m in [N-1-idx, N-1-idx+N-1] = [N-1-idx, 2N-2-idx]
    
    # The change: h[idx] -> -h[idx], so delta = -2*h[idx]
    # ac_full[m] changes by: delta * h[m - (N-1) - idx] + delta * h[idx - (m - (N-1))]
    #   = -2v * h[m - N + 1 - idx] - 2v * h[idx - m + N - 1]
    # with the convention that h[j]=0 if j not in [0,N-1]
    # Plus: if both terms refer to h[idx] itself (i.e., m-N+1-idx == idx AND idx-m+N-1 == idx),
    # we need to add back 4v^2 (since (-v)(-v) - (v)(v) = 0, but we subtracted 2v*v twice)
    # Actually: original contribution was v*h[idx], new is (-v)*(-v) = v^2.
    # Delta = v^2 - v^2 = 0? No wait.
    
    # Let me just do it numerically for the affected lags.
    # Affected lags: m where m - (N-1) - idx in [0,N-1] OR idx - (m-(N-1)) in [0,N-1]
    # i.e., m in [N-1+idx, N-1+idx+N-1] = [N-1+idx, 2N-2+idx]  (but m max is 2N-2)
    # AND m in [N-1-idx, N-1-idx+N-1] = [N-1-idx, 2N-2-idx]
    
    # Union: m in [N-1-idx, 2N-2+idx] intersect [0, 2N-2]
    # = [max(0, N-1-idx), min(2N-2, 2N-2+idx)] = [N-1-idx, 2N-2] since idx>=0
    
    m_lo = max(0, N - 1 - idx)
    m_hi = min(2*N - 2, 2*N - 2 + idx)
    
    # For each m in [m_lo, m_hi], compute delta_ac[m]
    # delta_ac[m] = -2v * h[m - (N-1) - idx] - 2v * h[idx - (m - (N-1))]
    # Let a = m - (N-1) - idx, b = idx - (m - (N-1)) = -(m - (N-1) - idx) = -a
    # So delta_ac[m] = -2v * (h[a] + h[-a]) with h[j]=0 if j not in [0,N-1]
    # But when a == 0 (i.e., m = N-1+idx), both terms are h[idx], so:
    #   original: v*v = v^2, new: (-v)*(-v) = v^2, delta = 0
    #   formula gives: -2v*(h[idx]+h[idx]) = -4v^2  -- WRONG
    # So we need: delta_ac[m] = -2v*h[a] - 2v*h[b] + 4v^2 * (1 if a==0 else 0)
    # Wait, when a==0: a=0, b=0, so h[a]=h[idx], h[b]=h[idx]
    # Original contribution to ac[m]: v*h[idx] (from i=idx, i+k=idx) -- but this is only counted once
    # Actually in the sum sum_i h[i]*h[i+k], when i=idx and i+k=idx, we need k=0, so m=N-1
    # Hmm, I'm overcomplicating this. Let me just use a simpler approach.
    
    # Simplest correct approach: the change to ac_full due to flipping h[idx] is:
    # For each lag m, the terms involving h[idx] are:
    #   h[idx] * h[idx + d] where d = m - (N-1), if 0 <= idx+d <= N-1
    #   h[idx + d'] * h[idx] where d' = (N-1) - m, if 0 <= idx+d' <= N-1
    # After flip, h[idx] -> -h[idx], so each such term changes sign.
    # delta_ac[m] = -2 * h[idx] * (sum of the OTHER h values paired with h[idx] at lag m)
    # If the "other" value is also h[idx] (i.e., d=0 and d'=0, meaning m=N-1 and idx is the center),
    # then the term h[idx]*h[idx] = v^2 becomes (-v)*(-v) = v^2, so delta = 0.
    # Otherwise, delta = -2v * (h[idx+d] + h[idx+d']) where applicable, with 0 for out-of-range.
    
    # Let d = m - (N-1)
    # Term 1: h[idx] * h[idx+d] valid if 0 <= idx+d <= N-1
    # Term 2: h[idx+d2] * h[idx] where d2 = -d, valid if 0 <= idx-d <= N-1
    # Note term1 and term2 are the same term when d=0 (i.e., m=N-1)
    
    # delta_ac[m]:
    #   if d != 0:
    #     t1 = h[idx+d] if 0<=idx+d<N else 0
    #     t2 = h[idx-d] if 0<=idx-d<N else 0
    #     delta = -2*v*(t1 + t2)
    #   if d == 0:
    #     delta = 0  (v^2 -> v^2)
    
    # This is O(1) per lag, and we need to check O(N) lags.
    # But we just need the new max|ac|, so we need to check all affected lags.
    
    # For efficiency, precompute the affected lags and their deltas.
    # m ranges from max(0, N-1-idx) to min(2N-2, N-1+idx)
    # Because d = m-(N-1), and we need |d| <= idx for h[idx±d] to be in range
    # (actually we need idx+d in [0,N-1] OR idx-d in [0,N-1])
    # idx+d in [0,N-1] means -idx <= d <= N-1-idx
    # idx-d in [0,N-1] means -(N-1-idx) <= d <= idx
    # Union: d in [-(N-1-idx), N-1-idx] = [-(N-1-idx), N-1-idx]
    # So m in [N-1-(N-1-idx), N-1+(N-1-idx)] = [idx, 2N-2-idx]
    
    m_lo = max(0, idx)
    m_hi = min(2*N - 2, 2*N - 2 - idx)
    
    # For each m in [m_lo, m_hi], compute new_ac[m] = ac[m] + delta_ac[m]
    # and track the max absolute value.
    
    # First, get the current max from unaffected lags
    # Unaffected: m < m_lo or m > m_hi
    if m_lo > 0:
        max_unaffected_lo = np.max(np.abs(ac[:m_lo])) if m_lo < len(ac) else 0.0
    else:
        max_unaffected_lo = 0.0
    if m_hi < len(ac) - 1:
        max_unaffected_hi = np.max(np.abs(ac[m_hi+1:])) if m_hi+1 < len(ac) else 0.0
    else:
        max_unaffected_hi = 0.0
    max_unaffected = max(max_unaffected_lo, max_unaffected_hi)
    
    # Now compute max over affected lags
    d_range = np.arange(m_lo - (N-1), m_hi - (N-1) + 1)  # d values
    # For each d, compute delta
    # t1: h[idx+d] valid if 0 <= idx+d < N
    # t2: h[idx-d] valid if 0 <= idx-d < N
    
    idx_plus_d = idx + d_range
    idx_minus_d = idx - d_range
    
    t1 = np.zeros(len(d_range))
    valid1 = (idx_plus_d >= 0) & (idx_plus_d < N)
    t1[valid1] = h[idx_plus_d[valid1]]
    
    t2 = np.zeros(len(d_range))
    valid2 = (idx_minus_d >= 0) & (idx_minus_d < N)
    t2[valid2] = h[idx_minus_d[valid2]]
    
    delta = -2.0 * v * (t1 + t2)
    # When d=0, the term is v^2 -> v^2, so delta should be 0, but our formula gives -4v^2
    # Fix: where d==0, set delta = 0
    zero_mask = (d_range == 0)
    if np.any(zero_mask):
        delta[zero_mask] = 0.0
    
    new_ac_affected = ac[m_lo:m_hi+1] + delta
    max_affected = np.max(np.abs(new_ac_affected))
    
    new_max = max(max_unaffected, max_affected)
    return float(new_max / nsq)

def _local_search_block(h, max_passes=50, deadline=None, block_sizes=(1, 2, 4, 8, 16)):
    """
    Local optimization: try flipping contiguous blocks of elements' signs.
    Much faster than single-flip search since each block flip is O(N) via FFT.
    """
    h = h.copy()
    cur_sum = float(np.sum(h))
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        best_improvement = 0
        best_action = None
        
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            for start in range(0, N - bs + 1, max(1, bs // 2)):
                if deadline is not None and time.time() > deadline:
                    break
                h_trial = h.copy()
                h_trial[start:start+bs] *= -1
                new_sum = cur_sum - 2.0 * np.sum(h_trial[start:start+bs]) + 2.0 * np.sum(h[start:start+bs])
                # Actually simpler: new_sum = cur_sum - 2*sum(flipped elements)
                new_sum = cur_sum - 2.0 * np.sum(h[start:start+bs])
                if abs(new_sum) < 1e-9:
                    continue
                new_val = _autoconv_max_abs(h_trial) / (new_sum * new_sum)
                improvement = best_val - new_val
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = (start, bs)
        
        if best_action is None or best_improvement < 1e-12:
            break
        
        start, bs = best_action
        h[start:start+bs] *= -1
        cur_sum = float(np.sum(h))
        best_val = _score(h)
    
    return h, best_val

def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local optimization: gradient-based coordinate descent using analytic gradient
    of the smooth surrogate.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    beta = 80.0
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            # Use small perturbations with line search
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _phase_optimize(h, max_iters=200, deadline=None):
    """
    CAZAC-inspired phase optimization: represent h as |h| * exp(i*phase),
    optimize phases to minimize peak-to-average power ratio in autoconvolution.
    Use gradient descent on the smooth surrogate.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    # Get magnitudes and phases
    H_fft = np.fft.rfft(h, 2*N)
    mags = np.abs(H_fft)
    phases = np.angle(H_fft)
    
    # Avoid zero magnitudes
    mags_safe = np.maximum(mags, 1e-10)
    
    beta = 100.0
    lr = 0.01
    rng = np.random.default_rng(42)
    
    best_h = h.copy()
    best_val = _score(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Reconstruct from current phases
        H_curr = mags_safe * np.exp(1j * phases)
        h_curr = np.fft.irfft(H_curr, 2*N)[:N]
        s_curr = np.sum(h_curr)
        if abs(s_curr) < 1e-9:
            break
        h_curr = h_curr / s_curr
        
        # Compute gradient of smooth objective w.r.t. phases
        # d/dphase_j [soft_max|ac| / s^2] 
        # Use numerical gradient for simplicity
        n_phase = len(phases)
        grad = np.zeros(n_phase)
        eps = 0.01
        
        for j in range(0, n_phase, max(1, n_phase // 50)):  # Subsample for speed
            if deadline is not None and time.time() > deadline:
                break
            phases_plus = phases.copy()
            phases_plus[j] += eps
            H_plus = mags_safe * np.exp(1j * phases_plus)
            h_plus = np.fft.irfft(H_plus, 2*N)[:N]
            s_plus = np.sum(h_plus)
            if abs(s_plus) > 1e-9:
                h_plus = h_plus / s_plus
                val_plus = _score_smooth(h_plus, beta)
            else:
                val_plus = 1e10
            
            phases_minus = phases.copy()
            phases_minus[j] -= eps
            H_minus = mags_safe * np.exp(1j * phases_minus)
            h_minus = np.fft.irfft(H_minus, 2*N)[:N]
            s_minus = np.sum(h_minus)
            if abs(s_minus) > 1e-9:
                h_minus = h_minus / s_minus
                val_minus = _score_smooth(h_minus, beta)
            else:
                val_minus = 1e10
            
            grad[j] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient step with adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            lr_adapt = lr * 0.5 if it % 20 == 0 else lr
            phases -= lr_adapt * grad / max(grad_norm, 1e-10)
        
        # Occasional random restart of phases
        if it % 50 == 0 and it > 0:
            phases = phases + rng.standard_normal(n_phase) * 0.1
        
        # Track best
        H_curr = mags_safe * np.exp(1j * phases)
        h_curr = np.fft.irfft(H_curr, 2*N)[:N]
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-9:
            h_curr = h_curr / s_curr
            val_curr = _score(h_curr)
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
        
        # Decay beta slowly for finer resolution
        beta *= 0.999
    
    return best_h, best_val

def _peak_suppression(h, basis, max_iters=50, deadline=None):
    """
    Peak suppression phase: directly target the argmax of the autoconvolution
    by adjusting parameters to reduce the peak value.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, _score(h)
    h = h / s
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(h)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        # Try to reduce the peak by adjusting parameters
        # Compute gradient of peak value w.r.t. parameters (approximate)
        # For simplicity, try small perturbations in parameter space
        params, _, _, _ = np.linalg.lstsq(basis.T, h, rcond=None)
        
        best_params = params.copy()
        best_val = _score(h)
        rng = np.random.default_rng(12345)
        
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            # Perturb parameters
            pert = rng.standard_normal(len(params)) * 0.1
            new_params = params + pert
            new_h = np.dot(new_params, basis)
            new_s = np.sum(new_h)
            if abs(new_s) < 1e-6:
                continue
            new_h = new_h / new_s
            new_val = _score(new_h)
            if new_val < best_val:
                best_val = new_val
                best_params = new_params
        
        h = np.dot(best_params, basis)
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        h = h / s
    
    return h, _score(h)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    n_basis = 25
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution (larger basis)
    if time.time() < deadline - 50:
        n_basis_de = min(40, N // 10)  # Use more basis functions for DE
        basis_de = _make_basis_functions(N, n_basis_de)
        bounds = [(-2.0, 2.0)] * n_basis_de
        
        def de_objective(params):
            if time.time() > deadline - 45:
                return 1e10
            h = np.dot(params, basis_de)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)
        
        try:
            res_de = differential_evolution(
                de_objective, bounds, 
                seed=42, maxiter=80, tol=1e-5,
                popsize=20, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            if res_de.fun < best_score:
                h_de = np.dot(res_de.x, basis_de)
                s_de = np.sum(h_de)
                if abs(s_de) > 1e-6:
                    h_de = h_de / s_de
                    val_de = _score(h_de)
                    if val_de < best_score:
                        best_score = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
        
        # Also try a direct random search in full N-space for diversity
        if time.time() < deadline - 40:
            rng_rs = np.random.default_rng(77)
            for _ in range(20):
                if time.time() > deadline - 38:
                    break
                h_rand = rng_rs.standard_normal(N)
                s_rand = np.sum(h_rand)
                if abs(s_rand) < 1e-6:
                    continue
                h_rand = h_rand / s_rand
                val_rand = _score(h_rand)
                if val_rand < best_score:
                    best_score = val_rand
                    best_h = h_rand.copy()
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate and adaptive beta
    if time.time() < deadline - 30:
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        if len(basis.shape) == 2:
            params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(8):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        # Adaptive beta schedule: 60 -> 120 -> 250
        beta_schedule = [60.0, 120.0, 250.0]
        
        for params_init in inits[:8]:
            if time.time() > deadline - 25:
                break
            
            try:
                current_params = params_init.copy()
                
                for beta in beta_schedule:
                    if time.time() > deadline - 23:
                        break
                    
                    def smooth_obj(params, b=beta):
                        if time.time() > deadline - 22:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, b)
                    
                    res_smooth = minimize(
                        smooth_obj, current_params, method='BFGS',
                        options={'maxiter': 150, 'disp': False, 'gtol': 1e-7}
                    )
                    current_params = res_smooth.x
                
                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > deadline - 20:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, current_params, method='L-BFGS-B',
                    options={'maxiter': 200, 'disp': False, 'gtol': 1e-8}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Peak suppression
    if time.time() < deadline - 18:
        try:
            h_ps, val_ps = _peak_suppression(best_h, basis, max_iters=30, deadline=deadline-16)
            if val_ps < best_score:
                best_score = val_ps
                best_h = h_ps.copy()
        except Exception:
            pass
    
    # Phase 4: Basin hopping
    if time.time() < deadline - 15:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > deadline - 12:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=15, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'gtol': 1e-7}},
                seed=42
            )
            if res_bh.fun < best_score:
                h_bh = np.dot(res_bh.x, basis)
                s_bh = np.sum(h_bh)
                if abs(s_bh) > 1e-6:
                    h_bh = h_bh / s_bh
                    val_bh = _score(h_bh)
                    if val_bh < best_score:
                        best_score = val_bh
                        best_h = h_bh.copy()
        except Exception:
            pass
    
    # Phase 5: Phase optimization (CAZAC-inspired)
    if time.time() < deadline - 12:
        try:
            h_phase, val_phase = _phase_optimize(best_h, max_iters=300, deadline=deadline-10)
            if val_phase < best_score:
                best_score = val_phase
                best_h = h_phase.copy()
        except Exception:
            pass
    
    # Phase 6: Block-flip local search (much faster than single-flip)
    if time.time() < deadline - 5:
        h_block, val_block = _local_search_block(best_h, max_passes=100, deadline=deadline-3)
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Phase 7: Final perturbation polish
    if time.time() < deadline - 1:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=5, deadline=deadline-0.3, step_size=0.02)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 1:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 0.3:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-10}
            )
            h_final = np.dot(res_final.x, basis)
            s_final = np.sum(h_final)
            if abs(s_final) > 1e-6:
                h_final = h_final / s_final
                val_final = _score(h_final)
                if val_final < best_score:
                    best_score = val_final
                    best_h = h_final.copy()
        except Exception:
            pass
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
