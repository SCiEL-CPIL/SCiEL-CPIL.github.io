"""Optimized PSL minimization for signed step function with N=400 intervals."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _legendre_sequence(n):
    """Generate a Legendre sequence of length n (quadratic residue mod p)."""
    # Find a prime p >= n
    p = n
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
    
    # Generate Legendre sequence
    seq = np.ones(n)
    for i in range(1, n):
        # Legendre symbol (i/p)
        if pow(i, (p - 1) // 2, p) == p - 1:
            seq[i] = -1.0
    return seq


def _m_sequence(n):
    """Generate an m-sequence (maximal length LFSR sequence) of length n."""
    # For simplicity, use a known good LFSR polynomial for lengths up to 400
    # This is a simplified approach - we'll use a random-like sequence with good properties
    # Actually, let's use a simpler deterministic pattern that has good autocorrelation
    # For now, use a pseudo-random sequence with fixed seed for reproducibility
    rng = np.random.default_rng(12345)
    return rng.choice([-1.0, 1.0], size=n)


def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    # Build the influence matrix S where S[i,k] = h[i-k] + h[i+k]
    S = np.zeros((n, n - 1), dtype=np.float64)
    for k in range(1, n):
        idx = k - 1
        # h[i-k] for i = k, k+1, ..., n-1
        S[k:, idx] += h[:n - k]
        # h[i+k] for i = 0, 1, ..., n-1-k
        S[:n - k, idx] += h[k:]
    
    # delta[i, k] = -2 * h[i] * S[i, k]
    delta = -2.0 * h[:, np.newaxis] * S
    
    # New acf if we flip position i: acf_new[k] = acf[k] + delta[i, k]
    acf_lags = acf[1:n]  # shape (n-1,)
    new_acf = delta + acf_lags[np.newaxis, :]
    new_psl = np.max(np.abs(new_acf), axis=1)
    
    # Find the i with minimum new_psl
    best_i = int(np.argmin(new_psl))
    best_new_psl = new_psl[best_i]
    
    return best_i, float(best_new_psl), current_psl


def _fast_psl_optimize(h, max_passes=100, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        # Check time
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if new_psl >= cur_psl:
            # No improvement possible with single flip
            break
        
        # Apply the flip
        h[i] = -h[i]
        passes += 1
        
        # Recompute acf
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        # If PSL reached 0, we're done
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl


def _simulated_annealing_psl(h, time_budget=30.0, start_time=None):
    """
    Simulated annealing on PSL using fast delta computation.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0  # initial temperature
    T_min = 0.1
    cooling = 0.95
    
    step = 0
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random position to flip
        i = np.random.randint(n)
        
        # Compute the PSL change from flipping position i
        delta = np.zeros(n - 1)
        for k in range(1, n):
            contrib = 0.0
            if i - k >= 0:
                contrib += h[i - k]
            if i + k < n:
                contrib += h[i + k]
            delta[k - 1] = -2.0 * h[i] * contrib
        
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            # Accept the move
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
        step += 1
    
    return best_h, best_psl


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the peak sidelobe level (upper bound on C3).
    
    Strategy:
    1. Try multiple initializations: Legendre sequence, m-sequence-like, random.
    2. For each, run fast gradient-based PSL optimization (Gordon's method).
    3. Use simulated annealing to escape local minima.
    4. Return the best sequence found.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Legendre sequence
    candidates.append(_legendre_sequence(N))
    
    # M-sequence-like (using a good deterministic pattern)
    rng0 = np.random.default_rng(999)
    candidates.append(rng0.choice([-1.0, 1.0], size=N))
    
    # Several random initializations
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed * 7)
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Phase 2: Fast gradient optimization from each candidate
    for h_init in candidates:
        if time.time() > deadline - 30:
            break
        
        h_opt, psl_opt = _fast_psl_optimize(
            h_init, 
            max_passes=100,
            time_budget=10.0,
            start_time=start_time
        )
        
        if psl_opt < best_score:
            best_score = psl_opt
            best_h = h_opt.copy()
    
    # Phase 3: Simulated annealing refinement on the best candidate
    if best_h is not None and time.time() < deadline - 5:
        h_sa, psl_sa = _simulated_annealing_psl(
            best_h,
            time_budget=25.0,
            start_time=start_time
        )
        
        if psl_sa < best_score:
            best_score = psl_sa
            best_h = h_sa.copy()
    
    # Phase 4: Final gradient polish if time permits
    if best_h is not None and time.time() < deadline - 2:
        h_final, psl_final = _fast_psl_optimize(
            best_h,
            max_passes=50,
            time_budget=3.0,
            start_time=start_time
        )
        
        if psl_final < best_score:
            best_score = psl_final
            best_h = h_final.copy()
    
    return best_h
# EVOLVE-BLOCK-END
