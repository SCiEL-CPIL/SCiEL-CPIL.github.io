import numpy as np
import time
import warnings

N = 50
_TIME_LIMIT = 145.0  # safety margin below 150s kill
_start_time = time.time()

def _autoconv_max(h):
    """Compute max of linear autoconvolution via FFT (fast)."""
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return np.max(conv)

def _score(h):
    """(sum h_i^2) / ((sum h_i) * max(autoconv))"""
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _score_grad(h):
    """
    Compute score and its gradient with respect to h.
    score = s2 / (s1 * M) where M = max autoconv.
    
    For the gradient, we need dM/dh_i. The autoconv at lag k is:
    (h * h)[k] = sum_j h_j * h_{j+k}
    
    The max M is achieved at some lag k*. 
    dM/dh_i = sum_j (delta_{i,j} * h_{j+k*} + h_j * delta_{i,j+k*})
            = h_{i+k*} + h_{i-k*} (with appropriate boundary handling)
    
    Actually for linear convolution:
    (h*h)[k] = sum_{j} h[j] * h[j+k] for k >= 0
    For k < 0, (h*h)[k] = sum_j h[j+k] * h[j] = (h*h)[|k|] by symmetry
    
    So M = max over all lags of (h*h)[|lag|].
    
    Let's compute the full autoconvolution and find where the max is.
    """
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0, np.zeros_like(h)
    
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    M = np.max(conv)
    if M < 1e-12:
        return 0.0, np.zeros_like(h)
    
    # Find where max occurs
    max_idx = np.argmax(conv)
    
    # The autoconvolution is symmetric: conv[k] = conv[size-1-k]
    # So we work with the non-negative lags: 0 to N-1
    # conv[k] for k in [0, N-1] is the standard autocorrelation at lag k
    # conv[k] for k in [N, 2N-2] mirrors conv[2N-2-k]
    
    # d(conv[k])/d(h[i]) = h[i+k] if i+k < N else 0, plus h[i-k] if i-k >= 0 else 0
    # But conv is symmetric, so conv[k] = conv[size-1-k]
    
    # Let's think in terms of the symmetric autocorrelation.
    # Let r[m] = sum_j h[j] * h[j+m] for m = 0, 1, ..., N-1
    # Then M = max(r[0], r[1], ..., r[N-1])
    
    # r[m] = sum_j h[j] * h[j+m]
    # d(r[m])/d(h[i]) = h[i+m] (if i+m < N) + h[i-m] (if i-m >= 0)
    
    # Find which lag gives the max
    r = np.zeros(N)
    for m in range(N):
        r[m] = np.dot(h[:N-m], h[m:N])
    
    m_star = np.argmax(r)
    M = r[m_star]
    
    # Gradient of M with respect to h[i]:
    # dM/dh[i] = h[i + m_star] (if i + m_star < N) + h[i - m_star] (if i - m_star >= 0)
    dM = np.zeros(N)
    for i in range(N):
        val = 0.0
        if i + m_star < N:
            val += h[i + m_star]
        if i - m_star >= 0:
            val += h[i - m_star]
        dM[i] = val
    
    # score = s2 / (s1 * M)
    # d(score)/d(h[i]) = [2*h[i]*s1*M - s2*(d(s1)/d(h[i])*M + s1*dM/d(h[i]))] / (s1*M)^2
    # d(s1)/d(h[i]) = 1
    # = [2*h[i]*s1*M - s2*(M + s1*dM[i])] / (s1*M)^2
    
    denom = (s1 * M) ** 2
    grad = np.zeros(N)
    for i in range(N):
        grad[i] = (2 * h[i] * s1 * M - s2 * (M + s1 * dM[i])) / denom
    
    score = s2 / (s1 * M)
    return score, grad

def _project_simplex(v, target_sum=None):
    """
    Project v onto the nonnegative orthant with sum = target_sum.
    If target_sum is None, preserve the current sum.
    """
    if target_sum is None:
        target_sum = np.sum(v)
    if target_sum <= 0:
        return np.zeros_like(v)
    
    # We want to find w >= 0 with sum(w) = target_sum, minimizing ||w - v||^2
    # This is equivalent to projecting v onto {w >= 0, sum(w) = target_sum}
    
    # Use the standard algorithm: find tau such that sum(max(v - tau, 0)) = target_sum
    # Sort v in descending order
    v_sorted = np.sort(v)[::-1]
    n = len(v)
    
    # Find the right tau
    # sum(max(v_i - tau, 0)) = target_sum
    # For a given tau, this is sum of (v_i - tau) for all v_i > tau
    
    # Try different numbers of active coordinates
    best_tau = None
    for k in range(n, 0, -1):
        # Assume top k coordinates are positive
        if k == 0:
            break
        tau = (np.sum(v_sorted[:k]) - target_sum) / k
        if tau < v_sorted[k - 1] + 1e-12:
            # Check that the k+1-th element is <= tau
            if k < n and v_sorted[k] > tau + 1e-12:
                continue
            best_tau = tau
            break
    
    if best_tau is None:
        # Fallback: all positive
        best_tau = (np.sum(v) - target_sum) / n
    
    w = np.maximum(v - best_tau, 0)
    # Renormalize to ensure exact sum
    if np.sum(w) > 0:
        w = w * (target_sum / np.sum(w))
    else:
        w = np.full(n, target_sum / n)
    
    return w

def _gradient_ascent(h, max_iters=2000, lr_init=0.01, lr_decay=0.999, tol=1e-12):
    """
    Gradient ascent on the score function, projecting onto the simplex.
    """
    h = h.copy()
    s1 = np.sum(h)
    if s1 < 1e-12:
        return h, 0.0
    
    best_score = _score(h)
    best_h = h.copy()
    lr = lr_init
    
    for it in range(max_iters):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        score, grad = _score_grad(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        # Gradient ascent step
        h_new = h + lr * grad
        
        # Project onto simplex with same sum
        h_new = _project_simplex(h_new, target_sum=s1)
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol:
            break
        
        h = h_new
        lr *= lr_decay
        
        # Adaptive learning rate: if score decreased, reduce lr
        if it % 50 == 0 and it > 0:
            current_score = _score(h)
            if current_score < best_score - 1e-8:
                lr *= 0.5
    
    final_score = _score(h)
    if final_score > best_score:
        best_score = final_score
        best_h = h.copy()
    
    return best_h, best_score

def _sidon_positions(n_pos):
    """
    Generate Sidon-like positions: positions where all pairwise differences are distinct.
    Use a greedy construction.
    """
    positions = [0]
    used_diffs = set()
    
    while len(positions) < n_pos:
        # Try next candidate position
        found = False
        for cand in range(positions[-1] + 1, N):
            ok = True
            for i in range(len(positions)):
                d = cand - positions[i]
                if d < 0 or d >= N:
                    continue
                if d in used_diffs:
                    ok = False
                    break
            if ok:
                # Add this position
                for i in range(len(positions)):
                    d = cand - positions[i]
                    if 0 <= d < N:
                        used_diffs.add(d)
                positions.append(cand)
                found = True
                break
        
        if not found:
            # Relax: allow some duplicate differences
            for cand in range(positions[-1] + 1, N):
                positions.append(cand)
                break
            break
    
    return positions[:n_pos]

def _structured_inits():
    """Generate diverse structured initializations."""
    inits = []
    
    # Sparse: few nonzero with equal weights
    for k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        h = np.zeros(N)
        h[:k] = 1.0 / k
        inits.append(h)
    
    # Sidon-like sparse
    for n_pos in [3, 4, 5, 6, 7, 8, 10]:
        positions = _sidon_positions(n_pos)
        h = np.zeros(N)
        for p in positions:
            h[p] = 1.0 / len(positions)
        inits.append(h)
    
    # Uneven Sidon-like
    for n_pos in [4, 5, 6, 7]:
        positions = _sidon_positions(n_pos)
        h = np.zeros(N)
        weights = np.linspace(0.5, 1.5, n_pos)
        weights /= weights.sum()
        for i, p in enumerate(positions):
            h[p] = weights[i]
        inits.append(h)
    
    # Spread: all nonzero
    h = np.ones(N) / N
    inits.append(h)
    
    # Gaussian-like
    idx = np.arange(N)
    center = N / 2
    h = np.exp(-0.5 * ((idx - center) / (N / 4)) ** 2)
    h /= h.sum()
    inits.append(h)
    
    # Two bumps
    h = np.zeros(N)
    h[5:10] = 0.2
    h[30:35] = 0.2
    h /= h.sum()
    inits.append(h)
    
    # Alternating
    h = np.zeros(N)
    h[::2] = 1.0 / (N // 2 + 1)
    inits.append(h)
    
    # Triangular
    h = np.abs(np.sin(np.pi * np.arange(N) / N))
    h /= h.sum()
    inits.append(h)
    
    # Exponential decay
    h = np.exp(-0.1 * np.arange(N))
    h /= h.sum()
    inits.append(h)
    
    # Power law
    h = 1.0 / (np.arange(N) + 1) ** 0.5
    h /= h.sum()
    inits.append(h)
    
    return inits

def _random_init(seed):
    """Generate a random nonnegative vector."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.1, 2.0, N)
    h = rng.dirichlet(alpha) * rng.uniform(0.5, 2.0)
    return h

def _sparse_random_init(seed, n_nonzero=None):
    """Generate a random sparse nonnegative vector."""
    rng = np.random.default_rng(seed)
    if n_nonzero is None:
        n_nonzero = rng.integers(3, 15)
    positions = rng.choice(N, size=n_nonzero, replace=False)
    h = np.zeros(N)
    weights = rng.dirichlet(np.ones(n_nonzero))
    h[positions] = weights
    return h

def _optimize_from_init(init, method='grad', max_iters=None):
    """Run optimization from an initialization."""
    h = np.array(init, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        return h, 0.0
    
    if method == 'grad':
        if max_iters is None:
            max_iters = 3000
        return _gradient_ascent(h, max_iters=max_iters)
    else:
        # Fallback: coordinate descent
        return _coordinate_descent(h, max_rounds=200)

def _coordinate_descent(h, max_rounds=200, tol=1e-10):
    """
    Coordinate descent on nonnegative simplex.
    """
    h = h.copy()
    best_score = _score(h)
    
    for rnd in range(max_rounds):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            old_val = h[i]
            mean_val = np.sum(h) / N
            candidates = [0.0, old_val]
            if old_val > 0:
                candidates += [old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5]
            candidates += [mean_val * 0.5, mean_val, mean_val * 1.5, mean_val * 2.0]
            candidates = [max(0.0, c) for c in candidates]
            candidates += [1e-4, 1e-3, 1e-2]
            
            best_cand = old_val
            best_cand_score = best_score
            
            for c in candidates:
                if abs(c - old_val) < 1e-12:
                    continue
                h[i] = c
                s = _score(h)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c
            
            h[i] = best_cand
            if best_cand != old_val:
                improved = True
                if best_cand_score > best_score:
                    best_score = best_cand_score
        
        if not improved:
            break
    
    return h, best_score

def _refine_1d(h, idx, n_trials=20):
    """Refine a single coordinate using grid search."""
    old_val = h[idx]
    s1 = np.sum(h)
    scale = max(s1 / N, 1e-3)
    lo = 0.0
    hi = 3.0 * scale
    
    best_val = old_val
    best_score = _score(h)
    
    grid = np.linspace(0, hi, n_trials)
    for g in grid:
        h[idx] = g
        s = _score(h)
        if s > best_score:
            best_score = s
            best_val = g
    
    h[idx] = best_val
    return best_val, best_score

def construct():
    best_h = None
    best_score = 0.0
    
    # Generate initializations
    inits = _structured_inits()
    
    # Add random inits
    for seed in range(15):
        inits.append(_random_init(seed))
    
    # Add sparse random inits
    for seed in range(20):
        inits.append(_sparse_random_init(seed))
    
    # Run gradient ascent from each init
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        h_opt, s_opt = _optimize_from_init(init, method='grad', max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Additional random restarts around best
    if best_h is not None:
        rng = np.random.default_rng(123)
        for _ in range(40):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.clip(h_pert, 0, None)
            h_opt, s_opt = _optimize_from_init(h_pert, method='grad', max_iters=1500)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    # Also try coordinate descent from best
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 10:
        h_opt, s_opt = _coordinate_descent(best_h, max_rounds=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Final polish with more gradient iterations
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 5:
        h_opt, s_opt = _gradient_ascent(best_h, max_iters=5000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.zeros(N)
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    
    return [float(x) for x in best_h]
