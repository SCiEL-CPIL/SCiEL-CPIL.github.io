"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x with itself."""
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(X * np.conj(X), n=size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Returns inf if sum(x) is 0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def _generate_legendre_sequence(L):
    """Generate a Legendre sequence of length L (should be prime)."""
    # Find next prime >= L
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    p = L
    while not is_prime(p):
        p += 1
    
    # Legendre sequence: 1 if i is quadratic residue mod p, -1 otherwise, 0 if i=0
    seq = np.zeros(p)
    for i in range(1, p):
        # Euler's criterion: i^((p-1)/2) mod p is 1 if residue, -1 otherwise
        if pow(i, (p - 1) // 2, p) == 1:
            seq[i] = 1.0
        else:
            seq[i] = -1.0
    seq[0] = 1.0  # Convention
    
    return seq[:L]

def _generate_msequence(L):
    """Generate an m-sequence of length L using a primitive polynomial."""
    # Find m such that 2^m - 1 >= L
    m = 1
    while (1 << m) - 1 < L:
        m += 1
    
    L_full = (1 << m) - 1
    
    # Primitive polynomials for small m
    polys = {
        2: [1, 0],
        3: [1, 0, 1],
        4: [1, 0, 0, 1],
        5: [1, 0, 0, 0, 1],
        6: [1, 0, 0, 1, 1, 1],
        7: [1, 0, 0, 0, 0, 1, 1],
        8: [1, 0, 0, 1, 0, 0, 0, 1],
        9: [1, 0, 0, 0, 0, 1, 1, 0, 1],
        10: [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        11: [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        12: [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        14: [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        15: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    }
    
    poly = polys.get(m, None)
    if poly is None:
        # Fallback
        poly = [1] + [0] * (m - 1) + [1]
    
    # Generate LFSR sequence
    state = [1] * m
    seq = []
    for _ in range(L_full):
        seq.append(1.0 if state[-1] else -1.0)
        fb = 0
        for i, p_bit in enumerate(poly):
            if p_bit:
                fb ^= state[i]
        state = [fb] + state[:-1]
    
    return np.array(seq[:L], dtype=np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def _score_from_fft(X, s):
        """Compute score from FFT and sum."""
        if abs(s) < 1e-12:
            return float('inf')
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2 * N - 1]
        return np.max(np.abs(ac)) / (s * s)

    # Phase 1: Generate diverse structured candidates
    candidates = []
    
    # All ones
    candidates.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # M-sequences with different seeds/lengths
    for m_len in [N, N + 1, N + 2, 511, 1023]:
        try:
            mseq = _generate_msequence(m_len)
            if len(mseq) >= N:
                candidates.append(mseq[:N].copy())
        except Exception:
            pass
    
    # Legendre sequences
    try:
        leg = _generate_legendre_sequence(N)
        candidates.append(leg)
    except Exception:
        pass
    
    # Barker-like: try to construct
    # Random ±1 with various seeds
    rng = np.random.default_rng(42)
    for seed in range(100):
        x = rng.choice([-1.0, 1.0], size=N)
        candidates.append(x)
    
    # Evaluate all candidates
    best_x = candidates[0]
    best_score = _score(best_x)
    
    for c in candidates[1:]:
        c = np.asarray(c, dtype=np.float64)
        if len(c) < N:
            continue
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Phase 2: Simulated Annealing with PAIRWISE SWAPS (preserves sum)
    x = best_x.copy()
    cur_score = best_score
    X = np.fft.rfft(x, fft_size)
    cur_sum = np.sum(x)
    
    # Precompute phase factors
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)
    
    def _score_from_fft(X, s):
        if abs(s) < 1e-12:
            return float('inf')
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2 * N - 1]
        return np.max(np.abs(ac)) / (s * s)
    
    # Simulated annealing with pairwise swaps
    T = 0.5  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.9995
    
    rng2 = np.random.default_rng(12345)
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        # Pick two random indices to swap
        idx1, idx2 = rng2.integers(0, N, size=2)
        if idx1 == idx2:
            continue
        
        # For pairwise swap to change the sequence, the two values must differ
        if x[idx1] == x[idx2]:
            continue
        
        # Swap: flip both bits
        delta1 = -2.0 * x[idx1]
        delta2 = -2.0 * x[idx2]
        
        # Sum is preserved (delta1 + delta2 = 0 since x[idx1] = -x[idx2])
        new_sum = cur_sum  # unchanged
        
        # Update FFT
        X_new = X + delta1 * exp_factors[idx1] + delta2 * exp_factors[idx2]
        
        new_score = _score_from_fft(X_new, new_sum)
        
        # Accept if better, or with probability exp(-(new_score - cur_score) / T)
        if new_score < cur_score:
            # Accept
            x[idx1] += delta1
            x[idx2] += delta2
            X = X_new
            cur_sum = new_sum
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            # Metropolis acceptance
            delta_e = new_score - cur_score
            if delta_e < 1e-15:
                accept = True
            else:
                accept_prob = np.exp(-delta_e / T)
                accept = rng2.random() < accept_prob
            
            if accept:
                x[idx1] += delta1
                x[idx2] += delta2
                X = X_new
                cur_sum = new_sum
                cur_score = new_score
            else:
                no_improve += 1
        
        # Cool down
        T = max(T_min, T * cooling_rate)
    
    # Phase 3: Final greedy sweep with single-bit flips (to adjust sum if beneficial)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            old_val = x[idx]
            new_val = -old_val
            delta = new_val - old_val
            
            X_new = X + delta * exp_factors[idx]
            new_sum = cur_sum + delta
            new_score = _score_from_fft(X_new, new_sum)
            
            if new_score < cur_score:
                x[idx] = new_val
                X = X_new
                cur_sum = new_sum
                cur_score = new_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
