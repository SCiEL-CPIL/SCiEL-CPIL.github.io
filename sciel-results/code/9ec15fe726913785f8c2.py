import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute max absolute periodic autocorrelation at non-zero lags, normalized by N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] is the zero-lag autocorrelation (= sum of squares = N for ±1)
    # We want max |R[k]| for k != 0
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _legendre_sequence(n):
    """Generate a Legendre sequence of length n (n should be prime or 1 mod 4)."""
    # Find the smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Generate Legendre symbols
    seq = np.ones(p, dtype=np.float64)
    for i in range(1, p):
        # Legendre symbol (i/p)
        ls = np.power(i, (p - 1) // 2, p)
        if ls == p - 1:  # -1 mod p
            seq[i] = -1
        # else it's 1 or 0 (i=0 gives 0, but we start from 1)
    
    # Trim or pad to length n
    if p >= n:
        return seq[:n]
    else:
        # Pad with +1
        result = np.ones(n, dtype=np.float64)
        result[:p] = seq
        return result

def _quadratic_residue_sequence(n):
    """Generate a quadratic residue (Barker-like) sequence."""
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    # Find smallest prime p ≡ 3 (mod 4) >= n
    p = n
    while not is_prime(p) or p % 4 != 3:
        p += 1
    
    seq = np.ones(p, dtype=np.float64)
    for i in range(1, p):
        ls = np.power(i, (p - 1) // 2, p)
        if ls == p - 1:
            seq[i] = -1
    
    if p >= n:
        return seq[:n]
    else:
        result = np.ones(n, dtype=np.float64)
        result[:p] = seq
        return result

def _m_sequence(n):
    """Generate an m-sequence (maximum length LFSR sequence) of length 2^m - 1, truncated to n."""
    # Find m such that 2^m - 1 >= n
    m = 1
    while (1 << m) - 1 < n:
        m += 1
    
    # Use a primitive polynomial for m-bit LFSR
    # For small m, use known primitive polynomials
    primitive_polys = {
        1: [1],
        2: [1, 0],
        3: [1, 1],
        4: [1, 0, 1],
        5: [1, 0, 1],
        6: [1, 0, 1],
        7: [1, 1, 0],
        8: [1, 0, 1],
        9: [1, 1, 0],
        10: [1, 0, 1],
    }
    
    if m not in primitive_polys:
        # Fallback: use a simple LFSR
        taps = [m-1, m-2]
    else:
        taps = primitive_polys[m]
    
    # Initialize LFSR with all 1s
    lfsr = (1 << m) - 1
    seq = []
    length = (1 << m) - 1
    
    for _ in range(length):
        seq.append(1.0 if (lfsr & 1) == 0 else -1.0)
        # Feedback
        feedback = 0
        for t in taps:
            feedback ^= (lfsr >> t) & 1
        lfsr = (lfsr >> 1) | (feedback << (m - 1))
    
    result = np.array(seq, dtype=np.float64)
    if len(result) >= n:
        return result[:n]
    else:
        padded = np.ones(n, dtype=np.float64)
        padded[:len(result)] = result
        return padded

def _local_search(h, max_iter, deadline, best_h, best_score):
    """Local search by flipping individual bits."""
    h = h.copy()
    current_score = _compute_metric(h)
    
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
    
    no_improve = 0
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try flipping a random position
        idx = np.random.randint(N)
        h[idx] *= -1
        new_score = _compute_metric(h)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            h[idx] *= -1  # Revert
            no_improve += 1
        
        if no_improve > 200:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Generate several candidate initial sequences
    candidates = []
    
    # 1. Legendre sequence
    try:
        candidates.append(_legendre_sequence(N))
    except:
        pass
    
    # 2. Quadratic residue sequence
    try:
        candidates.append(_quadratic_residue_sequence(N))
    except:
        pass
    
    # 3. M-sequence
    try:
        candidates.append(_m_sequence(N))
    except:
        pass
    
    # 4. Random ±1
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 5. Another random
    np.random.seed(123)
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates and pick the best
    best_h = None
    best_score = np.inf
    
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        if len(c) != N:
            continue
        s = _compute_metric(c)
        if s < best_score:
            best_score = s
            best_h = c.copy()
    
    if best_h is None:
        best_h = np.ones(N, dtype=np.float64)
        best_score = _compute_metric(best_h)
    
    # Local search with multiple restarts
    num_restarts = 5
    iters_per_restart = max(1, int((deadline - time.time()) / 5 / 0.001))
    # Cap iterations to avoid too many
    iters_per_restart = min(iters_per_restart, 5000)
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            h = best_h.copy()
        else:
            # Perturb best solution
            h = best_h.copy()
            n_flips = max(1, N // 10)
            flip_idx = np.random.choice(N, size=n_flips, replace=False)
            h[flip_idx] *= -1
        
        best_h, best_score = _local_search(h, iters_per_restart, deadline - 2, best_h, best_score)
    
    return best_h
