"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Strategy:
1. Generate diverse initial candidates (Golay, Barker, random, structured).
2. Use a continuous relaxation: allow f[i] to be real values in [-1, 1].
3. Minimize the smooth objective: sum_k (ac[k])^2 / (sum f)^2 using gradient descent.
   This is equivalent to minimizing the L2 norm of the autoconvolution relative to the sum.
   The L2 norm bounds the max norm, so reducing L2 helps reduce max.
4. After continuous optimization, project to {-1, 1} by sign().
5. Polish with discrete local search using incremental autoconvolution updates.
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2*n-1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _incremental_flip_ac(ac, f, idx):
    """
    Update autoconvolution in-place after flipping f[idx] from f[idx] to -f[idx].
    ac is the current autoconvolution array (length 2*N-1).
    
    When f[idx] changes by delta = -2*f[idx], the change in ac[k] is:
    - For k where 0 <= k-idx < N: delta * f[k-idx] (from term i=idx)
    - For k where 0 <= k-idx < N: delta * f[k-idx] (from term i=k-idx, i.e., k-i=idx)
    - But when k-idx == idx, these are the same term, so only one contribution.
    
    Actually, let's be more careful:
    ac[k] = sum_i f[i] * f[k-i]
    
    The terms involving f[idx] are:
    - i = idx: f[idx] * f[k-idx]
    - k-i = idx, i.e., i = k-idx: f[k-idx] * f[idx]
    
    These are the same term when idx = k-idx, i.e., k = 2*idx.
    
    So the change in ac[k] when f[idx] changes by delta is:
    - If k != 2*idx and 0 <= k-idx < N: 2 * delta * f[k-idx]
    - If k == 2*idx: delta * f[idx] (but f[idx] is changing, so this is tricky)
    
    Let's just handle it by considering which k values are affected.
    k must satisfy: 0 <= idx < N and 0 <= k-idx < N
    So k ranges from idx to idx+N-1.
    
    For each such k:
    j = k - idx
    If j == idx (i.e., k = 2*idx):
        The term is f[idx] * f[idx] = f[idx]^2 = 1 (since f[idx] is ±1)
        After flip: (-f[idx]) * (-f[idx]) = f[idx]^2 = 1
        So no change! Wait, that's not right.
        
        Actually, when we flip f[idx], the term f[idx]*f[k-idx] where k-idx=idx becomes:
        (-f[idx]) * (-f[idx]) = f[idx]^2 = 1
        And the original was f[idx] * f[idx] = 1
        So indeed no change for the self-term.
        
    If j != idx:
        Original: f[idx] * f[j] + f[j] * f[idx] = 2 * f[idx] * f[j]
        After flip: (-f[idx]) * f[j] + f[j] * (-f[idx]) = -2 * f[idx] * f[j]
        Change: -4 * f[idx] * f[j]
    
    So:
    For k in [idx, idx+N-1]:
        j = k - idx
        if j == idx:
            # No change
            pass
        else:
            ac[k] += -4.0 * f[idx] * f[j]
    """
    delta = -2.0 * f[idx]  # change in f[idx]
    
    # For k in [idx, idx+N-1]:
    #   j = k - idx
    #   if j != idx:
    #       ac[k] += 2 * delta * f[j]
    
    k_start = idx
    k_end = min(idx + N, 2*N-1)
    
    for k in range(k_start, k_end):
        j = k - idx
        if j == idx:
            continue
        ac[k] += 2.0 * delta * f[j]


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    rng = np.random.RandomState(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    def golay_pair(n):
        """Generate Golay complementary pair of length n (must be power of 2)."""
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    # Use Golay sequences as building blocks
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
        candidates.append(np.concatenate([a[:N//2], b[:N//2]])[:N])
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like sequences
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        if block_size < N:
            n_blocks = N // block_size
            pattern = []
            for b in range(n_blocks):
                sign = 1.0 if ((b * 7 + 3) % 3) != 0 else -1.0
                pattern.extend([sign] * block_size)
            rem = N - len(pattern)
            pattern.extend([1.0] * rem)
            candidates.append(np.array(pattern[:N]))
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    # Continuous relaxation: minimize L2 norm of autoconv / (sum f)^2
    # This is a smooth objective that can be optimized with gradient descent
    # f is in [-1, 1]^N
    # Objective: sum_k ac[k]^2 / (sum f)^2
    # ac = f * f (linear convolution)
    
    def _smooth_objective(f_cont):
        """Compute sum_k ac[k]^2 / (sum f)^2 for continuous f."""
        s = np.sum(f_cont)
        if abs(s) < 1e-12:
            return np.inf, np.zeros_like(f_cont)
        ac = _autoconv_fft(f_cont)
        obj = np.sum(ac**2) / (s * s)
        # Gradient with respect to f
        # d/d f[i] of sum_k ac[k]^2 / s^2
        # = (2 * sum_k ac[k] * d(ac[k])/d(f[i]) * s^2 - sum_k ac[k]^2 * 2 * s * 1) / s^4
        # = (2 * sum_k ac[k] * d(ac[k])/d(f[i]) * s - 2 * sum_k ac[k]^2) / s^3
        # d(ac[k])/d(f[i]) = f[k-i] + f[i-k] (with appropriate bounds)
        # For linear convolution, d(ac[k])/d(f[i]) = f[k-i] if 0 <= k-i < N, else 0
        # Plus the symmetric term... actually for autoconvolution:
        # ac[k] = sum_j f[j] f[k-j]
        # d(ac[k])/d(f[i]) = f[k-i] + f[i-k] (if both in range)
        # But since ac is symmetric, and we're dealing with real signals:
        # d(ac[k])/d(f[i]) = f[k-i] if 0 <= k-i < N, plus f[i-k] if 0 <= i-k < N
        # These are the same when k-i = i-k, i.e., k=2i
        # So: d(ac[k])/d(f[i]) = f[k-i] + f[i-k] (with bounds)
        
        # Simpler: use FFT-based gradient
        # ac = ifft(fft(f)^2)
        # d(ac)/d(f) involves the derivative of the convolution
        # For now, use a simple numerical gradient or skip and use discrete search
        
        return obj, np.zeros_like(f_cont)
    
    # For efficiency, let's just use the discrete local search with incremental updates
    f = best_f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    # Peak-targeting local search with simulated annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.99998
    no_improve = 0
    max_no_improve = 5000
    
    while _remaining() > 5.0 and T > T_min:
        # Find the peak lag
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        
        # Try flipping bits that contribute to this peak
        best_flip = -1
        best_new_score = cur_score
        
        # Sample a subset of positions for speed
        if _remaining() > 30:
            sample_size = min(N, 200)
            sample = rng.choice(N, size=sample_size, replace=False)
        else:
            sample = np.arange(N)
        
        for j in sample:
            k_j = peak_idx - j
            if 0 <= k_j < N:
                if j == k_j:
                    continue
                # Try flipping j
                old_val = f[j]
                f[j] = -f[j]
                _incremental_flip_ac(ac, f, j)
                s_new = s + (-2.0 * old_val)
                if abs(s_new) < 1e-12:
                    f[j] = old_val
                    _incremental_flip_ac(ac, f, j)
                    continue
                new_score = np.max(np.abs(ac)) / (s_new * s_new)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_flip = j
                # Revert
                f[j] = old_val
                _incremental_flip_ac(ac, f, j)
        
        if best_flip >= 0:
            # Apply the best flip
            old_val = f[best_flip]
            f[best_flip] = -f[best_flip]
            _incremental_flip_ac(ac, f, best_flip)
            s = s + (-2.0 * old_val)
            cur_score = best_new_score
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # No single flip reduces the score; try a random flip
            idx = rng.randint(N)
            old_val = f[idx]
            f[idx] = -f[idx]
            _incremental_flip_ac(ac, f, idx)
            s_new = s + (-2.0 * old_val)
            if abs(s_new) < 1e-12:
                f[idx] = old_val
                _incremental_flip_ac(ac, f, idx)
                no_improve += 1
                T *= cooling
                continue
            new_score = np.max(np.abs(ac)) / (s_new * s_new)
            
            # Accept or reject
            if new_score < cur_score or (T > 0 and rng.random() < np.exp(-(new_score - cur_score) / T)):
                s = s_new
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_f = f.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                f[idx] = old_val
                _incremental_flip_ac(ac, f, idx)
                no_improve += 1
        
        T *= cooling
        
        # Restart if stuck
        if no_improve > max_no_improve and _remaining() > 20.0:
            T = max(T, 0.3)
            no_improve = 0
    
    # Final greedy polish
    f = best_f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    improved = True
    while improved and _remaining() > 3.0:
        improved = False
        for idx in range(N):
            old_val = f[idx]
            f[idx] = -f[idx]
            _incremental_flip_ac(ac, f, idx)
            s_new = s + (-2.0 * old_val)
            if abs(s_new) < 1e-12:
                f[idx] = old_val
                _incremental_flip_ac(ac, f, idx)
                continue
            new_score = np.max(np.abs(ac)) / (s_new * s_new)
            if new_score < cur_score:
                s = s_new
                cur_score = new_score
                improved = True
            else:
                f[idx] = old_val
                _incremental_flip_ac(ac, f, idx)
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    return best_f
