"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _calc_acf(s):
    """Compute the full autocorrelation of a ±1 sequence.
    Returns array A where A[k] = sum_i s[i]*s[i+k] for k in 0..N-1.
    """
    # Use FFT for efficiency
    n = len(s)
    f = np.fft.rfft(s, n=2 * n)
    acf_full = np.fft.irfft(f * np.conj(f), n=2 * n)
    # acf_full[0..n-1] are the positive lags
    return acf_full[:n]


def _peak_sidelobe(s):
    """Return max |A[k]| for k >= 1 (sidelobes only)."""
    acf = _calc_acf(s)
    if len(acf) <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))


def _evaluate_flip(s, i, acf_cache=None):
    """Evaluate the change in peak sidelobe if we flip bit i.
    Flipping s[i] changes the autocorrelation in a structured way.
    For a ±1 sequence, flipping s[i] from +1 to -1 (or vice versa)
    changes A[k] by -2*s[i]*s[i+k] for each lag k (with appropriate indexing).
    
    Actually, let's think more carefully. If s is the current sequence and we flip index i:
    s' = s with s[i] -> -s[i]
    A'[k] = sum_j s'[j]*s'[j+k]
    = sum_{j != i, j+k != i} s[j]*s[j+k] + s'[i]*s'[i+k] + s'[i-k]*s'[i] - s[i]*s[i] (if i and i+k are both in range)
    
    This is complex. Let's just recompute ACF for candidate flips in a batched way,
    or use the fact that flipping one bit changes ACF in a simple manner.
    
    For flipping index i:
    A'[k] = A[k] - 2*s[i]*s[i+k] for k where both i and i+k are valid, and i != i+k
    More precisely:
    A[k] = sum_j s[j] s[j+k]
    After flip at i: s'[j] = s[j] for j != i, s'[i] = -s[i]
    A'[k] = sum_j s'[j] s'[j+k]
    Terms involving index i: j=i or j+k=i
    If j=i: term is s'[i]*s'[i+k] = (-s[i])*s[i+k] = -s[i]*s[i+k] (if i+k in range and i+k != i)
    If j+k=i, i.e., j=i-k: term is s'[i-k]*s'[i] = s[i-k]*(-s[i]) = -s[i-k]*s[i] (if i-k in range and i-k != i)
    If j=i and j+k=i simultaneously, i.e., k=0: term is s'[i]*s'[i] = s[i]^2 = 1 (unchanged)
    
    So for k > 0:
    A'[k] = A[k] - s[i]*s[i+k] - s[i-k]*s[i] + (-s[i]*s[i+k]) + (-s[i-k]*s[i]) ... 
    Wait, let me be more careful.
    
    Original: A[k] = sum_j s[j] s[j+k]
    The terms where j=i: s[i] s[i+k]
    The terms where j+k=i, i.e., j=i-k: s[i-k] s[i]
    
    After flip:
    j=i term becomes: s'[i] s'[i+k] = (-s[i]) s[i+k] = -s[i] s[i+k]
    j=i-k term becomes: s'[i-k] s'[i] = s[i-k] (-s[i]) = -s[i-k] s[i]
    
    So A'[k] = A[k] - s[i]s[i+k] - s[i-k]s[i] + (-s[i]s[i+k]) + (-s[i-k]s[i])
             = A[k] - 2 s[i] s[i+k] - 2 s[i-k] s[i]
    
    But we need to be careful about boundaries and whether i+k or i-k are in range.
    Also, if i+k == i (k=0), it's a different case.
    
    For simplicity and correctness, let's just recompute.
    """
    s_new = s.copy()
    s_new[i] = -s_new[i]
    acf_new = _calc_acf(s_new)
    return float(np.max(np.abs(acf_new[1:])))


def _greedy_polish(s, max_iters=50000, time_budget=8.0):
    """Greedy local search: flip the bit that most reduces peak sidelobe."""
    t0 = time.time()
    s = s.copy()
    current_peak = _peak_sidelobe(s)
    
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        best_delta = 0
        best_i = -1
        # Sample a subset of positions for speed
        if N > 200:
            candidates = np.random.choice(N, size=min(N, 100), replace=False)
        else:
            candidates = np.arange(N)
        
        for i in candidates:
            peak = _evaluate_flip(s, i)
            delta = peak - current_peak
            if delta < best_delta:
                best_delta = delta
                best_i = i
        
        if best_i >= 0 and best_delta < -1e-12:
            s[best_i] = -s[best_i]
            current_peak += best_delta
        else:
            # Try a few random flips to escape local minima
            for _ in range(5):
                i = np.random.randint(N)
                peak = _evaluate_flip(s, i)
                if peak < current_peak - 1e-12:
                    s[i] = -s[i]
                    current_peak = peak
                    break
            else:
                # Stuck, do a random restart of a few bits
                for _ in range(3):
                    i = np.random.randint(N)
                    s[i] = -s[i]
                current_peak = _peak_sidelobe(s)
                break
    
    return s, _peak_sidelobe(s)


def _continuous_seed(time_budget=15.0):
    """Use continuous gradient descent on L2 norm of ACF to get a good binary seed."""
    t0 = time.time()
    np.random.seed(42)
    
    # Start with random ±1
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Gradient descent on sum_k A[k]^2 for k>=1
    lr = 0.05
    for it in range(2000):
        if time.time() - t0 > time_budget:
            break
        acf = _calc_acf(x)
        # Gradient of sum A[k]^2 w.r.t. x[i]
        # d/dx[i] sum_k (sum_j x[j]x[j+k])^2
        # = 2 * sum_k A[k] * dA[k]/dx[i]
        # dA[k]/dx[i] = x[i+k] + x[i-k] (for valid indices)
        grad = np.zeros(N)
        for k in range(1, N):
            # dA[k]/dx[i] = x[i+k] (if i+k < N) + x[i-k] (if i-k >= 0)
            idx_plus = np.arange(N - k)
            idx_minus = np.arange(k, N)
            grad[idx_plus] += 2 * acf[k] * x[idx_plus + k]
            grad[idx_minus] += 2 * acf[k] * x[idx_minus - k]
        
        x -= lr * grad
        # Project back to ±1
        x = np.where(x > 0, 1.0, -1.0)
    
    return x.astype(np.int8)


def _anneal(s, time_budget=20.0):
    """Simulated annealing to polish the sequence."""
    t0 = time.time()
    s = s.copy().astype(np.float64)
    current_peak = _peak_sidelobe(s)
    best_s = s.copy()
    best_peak = current_peak
    
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - t0 < time_budget:
        i = np.random.randint(N)
        s[i] = -s[i]
        new_peak = _peak_sidelobe(s)
        delta = new_peak - current_peak
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_s = s.copy()
        else:
            s[i] = -s[i]
        
        T *= cooling
    
    return best_s.astype(np.int8), best_peak


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    np.random.seed(12345)
    
    best_s = None
    best_peak = np.inf
    
    # Phase 1: Continuous optimization for a good initial seed
    if time.time() < deadline - 60.0:
        seed_s = _continuous_seed(time_budget=15.0)
        seed_peak = _peak_sidelobe(seed_s)
        best_s = seed_s.copy()
        best_peak = seed_peak
    
    # Phase 2: Greedy polish
    if time.time() < deadline - 30.0 and best_s is not None:
        polished, peak = _greedy_polish(best_s, max_iters=30000, time_budget=10.0)
        if peak < best_peak:
            best_s = polished
            best_peak = peak
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15.0 and best_s is not None:
        annealed, peak = _anneal(best_s, time_budget=15.0)
        if peak < best_peak:
            best_s = annealed
            best_peak = peak
    
    # Phase 4: Final greedy polish
    if time.time() < deadline - 5.0 and best_s is not None:
        final, peak = _greedy_polish(best_s, max_iters=20000, time_budget=8.0)
        if peak < best_peak:
            best_s = final
            best_peak = peak
    
    if best_s is None:
        # Fallback: all ones
        return np.ones(N, dtype=np.int8)
    
    return best_s

# EVOLVE-BLOCK-END
