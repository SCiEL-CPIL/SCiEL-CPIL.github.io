import numpy as np
import time

N = 400
DX = 0.5 / N

def _linear_acf(h):
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    acf_full = np.fft.irfft(power, n=fft_size)
    acf = np.empty(2 * n - 1, dtype=np.float64)
    acf[n-1] = acf_full[0]
    for k in range(1, n):
        acf[n-1+k] = acf_full[k]
        acf[n-1-k] = acf_full[fft_size - k]
    return acf

def _score(h):
    acf = _linear_acf(h)
    n = len(h)
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    max_abs = max(np.max(np.abs(right)), np.max(np.abs(left)))
    return float(DX * max_abs)

def _psl_value(acf, n):
    if n <= 1:
        return 0.0
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    return float(max(np.max(np.abs(right)), np.max(np.abs(left))))

def _compute_acf(h):
    return _linear_acf(h)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = [1.0]
    while len(seq) < n:
        seq.extend([-x for x in seq])
    return np.array(seq[:n])

def _legendre_sequence(n, p):
    """Generate Legendre sequence of length n using prime p > n."""
    seq = np.zeros(n)
    for i in range(1, n):
        # Legendre symbol (i/p)
        ls = pow(i, (p-1)//2, p)
        if ls == p-1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0
    seq[0] = 1.0
    return seq

def _barker_like(n):
    """Generate Barker-like sequence (maximal length 13 for binary)."""
    barkers = [
        [1,1,1,-1,1,-1,1],
        [1,1,1,-1,1,1,-1],
    ]
    base = np.array(barkers[0])
    seq = np.zeros(n)
    for i in range(n):
        seq[i] = base[i % len(base)]
        if (i // len(base)) % 2 == 1:
            seq[i] = -seq[i]
    return seq

def _m_sequence(n, degree, poly):
    """Generate m-sequence (maximal length LFSR sequence)."""
    m = (1 << degree) - 1
    state = 1
    seq = np.zeros(n)
    for i in range(n):
        seq[i] = 1.0 if (state & 1) else -1.0
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly
        state &= m
    return seq

def _structured_candidates():
    """Generate diverse structured candidates."""
    candidates = []
    
    tm = _thue_morse(N)
    candidates.append(tm)
    candidates.append(-tm)
    for shift in range(1, 8):
        candidates.append(np.roll(tm, shift))
        candidates.append(np.roll(-tm, shift))
    
    primes = [409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    for p in primes:
        if p > N:
            leg = _legendre_sequence(N, p)
            candidates.append(leg)
            candidates.append(-leg)
    
    polys = [
        0x0130,
        0x0160,
        0x0190,
        0x01C0,
        0x0230,
    ]
    for poly in polys:
        mseq = _m_sequence(N, 9, poly)
        candidates.append(mseq)
        candidates.append(-mseq)
        for shift in range(1, 5):
            candidates.append(np.roll(mseq, shift))
    
    candidates.append(_barker_like(N))
    candidates.append(-_barker_like(N))
    
    t = np.arange(N, dtype=np.float64)
    for root in [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        val = np.cos(2.0 * np.pi * root * t * t / N)
        candidates.append(np.sign(val))
    
    rng = np.random.default_rng(123)
    for trial in range(30):
        alpha = rng.uniform(0.01, 0.5)
        beta = rng.uniform(0.001, 0.1)
        gamma = rng.uniform(0.0, 2.0 * np.pi)
        delta = rng.uniform(-0.5, 0.5)
        val = (np.cos(alpha * t * t)
               + beta * np.cos(2.0 * np.pi * t / N + gamma)
               + delta)
        candidates.append(np.sign(val))
    
    alphas = [np.sqrt(2), np.sqrt(3), (1 + np.sqrt(5)) / 2, np.sqrt(5)]
    phases = [0.0, np.pi / 4, np.pi / 3, np.pi / 2]
    for alpha in alphas:
        for phase in phases:
            vals = np.sin(np.pi * alpha * t + phase)
            signs = np.sign(vals)
            signs[signs == 0] = 1.0
            candidates.append(signs)
    
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    for period in [3, 5, 7, 11, 13, 17]:
        pattern = np.sign(np.cos(2.0 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
    
    for i, c in enumerate(candidates):
        if np.all(c == 0):
            candidates[i] = np.ones(N)
    
    return candidates

def _fast_psl_optimize(h, time_budget, start_time, max_steps=100000, lr_init=0.5, lr_decay=0.9999):
    """
    Fast PSL optimization using FFT-based gradient computation.
    Uses epigraph formulation with auxiliary variable t.
    """
    n = len(h)
    h = h.copy().astype(np.float64)
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    # Initialize with small positive values for epigraph
    t = _psl_value(_compute_acf(h), n) + 0.1
    t = max(t, 0.1)
    
    best_h = h.copy()
    best_score = _score(h)
    lr = lr_init
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute ACF via FFT
        H = np.fft.rfft(h, n=fft_size)
        power = np.abs(H) ** 2
        acf_full = np.fft.irfft(power, n=fft_size)
        
        # Extract lagged ACF values
        acf = np.empty(2 * n - 1, dtype=np.float64)
        acf[n-1] = acf_full[0]
        for k in range(1, n):
            acf[n-1+k] = acf_full[k]
            acf[n-1-k] = acf_full[fft_size - k]
        
        # Compute PSL value
        center = n - 1
        right = acf[center+1:]
        left = acf[:center]
        psl = max(np.max(np.abs(right)), np.max(np.abs(left)))
        
        # Compute gradient of PSL with respect to h
        # d/dh_i [acf[k]] = -2 * sum over j of h[j] * delta(i-j-k) - 2 * sum over j of h[j] * delta(i+j-k)
        # More efficiently: use FFT to compute gradient
        grad = np.zeros(n, dtype=np.float64)
        
        # Find the lag(s) that achieve the maximum
        max_right_idx = np.argmax(np.abs(right))
        max_left_idx = np.argmax(np.abs(left))
        
        max_right_val = right[max_right_idx]
        max_left_val = left[max_left_idx]
        
        # Determine which lag dominates
        if abs(max_right_val) >= abs(max_left_val):
            dominant_lag = max_right_idx + 1
            dominant_sign = np.sign(max_right_val)
        else:
            dominant_lag = -(max_left_idx + 1)
            dominant_sign = np.sign(max_left_val)
        
        # Gradient contribution from dominant lag
        # d/dh_i [acf[dominant_lag]] involves correlations with shift dominant_lag
        if dominant_lag > 0:
            # acf[dominant_lag] = sum_j h[j] * h[j+dominant_lag]
            # d/dh_i = h[i+dominant_lag] + h[i-dominant_lag] (with boundary handling)
            for i in range(n):
                term1 = h[i + dominant_lag] if i + dominant_lag < n else 0.0
                term2 = h[i - dominant_lag] if i - dominant_lag >= 0 else 0.0
                grad[i] += dominant_sign * (term1 + term2)
        else:
            lag = -dominant_lag
            for i in range(n):
                term1 = h[i + lag] if i + lag < n else 0.0
                term2 = h[i - lag] if i - lag >= 0 else 0.0
                grad[i] += dominant_sign * (term1 + term2)
        
        # Update h in direction of negative gradient, then project to ±1
        h_new = h - lr * grad
        h_new = np.sign(h_new)
        h_new[h_new == 0] = 1.0
        
        # Check if improvement
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
        
        # Update h for next iteration (accept even if not best, for exploration)
        h = h_new
        
        # Decay learning rate
        lr *= lr_decay
        if lr < 0.01:
            lr = 0.01
        
        # Early stopping if converged
        if best_score < 0.08 * DX:
            break
    
    return best_h, best_score

def _local_search(h, time_budget, start_time, rng, max_iter=5000):
    """
    Local search with batch flips and momentum.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        delta = np.zeros(2 * n - 1, dtype=np.float64)
        for k in range(1, n):
            term1 = 0.0
            term2 = 0.0
            if i + k < n:
                term1 = h[i + k]
            if i - k >= 0:
                term2 = h[i - k]
            delta[k] = -2.0 * h[i] * (term1 + term2)
            delta[-k] = delta[k]
        
        new_acf = acf + delta
        new_psl = _psl_value(new_acf, n)
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or (current_psl > 0 and rng.random() < np.exp(-delta_psl / max(0.1, current_psl * 0.1))):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        if iteration % 100 == 50:
            num_flips = int(rng.integers(3, 6))
            indices = rng.choice(n, size=num_flips, replace=False)
            
            batch_delta = np.zeros(2 * n - 1, dtype=np.float64)
            for idx in indices:
                for k in range(1, n):
                    term1 = 0.0
                    term2 = 0.0
                    if idx + k < n:
                        term1 = h[idx + k]
                    if idx - k >= 0:
                        term2 = h[idx - k]
                    batch_delta[k] += -2.0 * h[idx] * (term1 + term2)
                    batch_delta[-k] += batch_delta[k]
            
            new_acf_batch = acf + batch_delta
            new_psl_batch = _psl_value(new_acf_batch, n)
            
            if new_psl_batch < current_psl:
                for idx in indices:
                    h[idx] = -h[idx]
                acf = _compute_acf(h)
                current_psl = _psl_value(acf, n)
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(42)
    
    candidates = _structured_candidates()
    
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])
    
    # Use fast PSL optimizer on top candidates
    top_k = 10
    for i, (s, h) in enumerate(scored[:top_k]):
        if time.time() > deadline - 30:
            break
        time_limit = min(deadline - 20, time.time() + 10)
        h_refined, s_refined = _fast_psl_optimize(
            h, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Random restarts with fast PSL optimizer
    while time.time() < deadline - 15:
        if rng.random() < 0.5:
            h_init = rng.choice([-1.0, 1.0], size=N)
        else:
            base = scored[rng.integers(min(top_k, len(scored)))][1]
            h_init = base.copy()
            num_flips = rng.integers(5, 20)
            flip_idx = rng.choice(N, size=num_flips, replace=False)
            h_init[flip_idx] *= -1
        
        time_limit = min(deadline - 10, time.time() + 8)
        h_refined, s_refined = _fast_psl_optimize(
            h_init, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final polish with local search
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(15.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 3.0:
            h_polished, s_polished = _local_search(
                best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
