"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return float(np.max(np.abs(ac)))


def _score(h):
    """Compute max|autoconv| / (sum h)^2. Lower is better."""
    s = float(np.sum(h))
    if abs(s) < 1e-9:
        return 1e15
    return _autoconv_max_abs(h) / (s * s)


def _legendre_sequence(n):
    """Generate Legendre symbol sequence of length n (quadratic residue mod p)."""
    p = n
    while p < n:
        p += 1
    # Find next prime >= n
    def is_prime(k):
        if k < 2:
            return False
        if k == 2:
            return True
        if k % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(k)) + 1, 2):
            if k % i == 0:
                return False
        return True
    while not is_prime(p):
        p += 1
    x = np.zeros(n)
    for i in range(1, n):
        # Legendre symbol (i|p)
        x[i] = np.sign(pow(i, (p - 1) // 2, p) - (p - pow(i, (p - 1) // 2, p)))
    x[0] = 1.0
    return x


def _m_sequence(n, m=8):
    """Generate an m-sequence (maximal length LFSR sequence) of length n."""
    # Use a known primitive polynomial for m=8: x^8 + x^4 + x^3 + x^2 + 1
    # For simplicity, use a simpler approach: generate from a linear recurrence
    # Use m=7: x^7 + x^3 + 1 (period 127)
    # For n=400, we'll tile or truncate
    # Simpler: use a de Bruijn-like or just a good LFSR
    # Let's use m=9: x^9 + x^5 + 1 (period 511)
    # Actually, let's just generate a pseudo-random binary sequence with good properties
    # Use a simple LFSR with polynomial x^8 + x^6 + x^5 + x^4 + 1
    # For robustness, generate multiple and pick best
    best_x = None
    best_val = 1e15
    for seed in range(20):
        rng = np.random.default_rng(seed)
        # Generate a random binary sequence and check autocorrelation
        x = rng.choice([-1.0, 1.0], size=n)
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    return best_x


def _barker_like(n):
    """Generate a Barker-like sequence (low autocorrelation)."""
    # Barker sequences of length 13: 1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1
    # Tile and modify for length n
    bark = np.array([1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
    x = np.tile(bark, (n // len(bark) + 1))[:n]
    return x


def _local_search_full_pass(h, max_passes=30, deadline=None):
    """
    Local search: in each pass, try flipping each element.
    Keep flips that improve the score. Multiple passes until no improvement.
    """
    h = h.copy()
    best_val = _score(h)
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            orig = h[i]
            h[i] = -orig
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                h[i] = orig
        if not improved:
            break
    
    return h, best_val


def _local_search_sparse(h, max_passes=20, deadline=None):
    """
    Local search allowing 0, ±1 values.
    Try: flip sign, set to 0, set from 0 to ±1.
    """
    h = h.copy()
    best_val = _score(h)
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping non-zero elements
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] != 0:
                orig = h[i]
                h[i] = -orig
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
        
        # Try setting non-zero to zero
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] != 0:
                orig = h[i]
                h[i] = 0.0
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
        
        # Try setting zero to ±1
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] == 0:
                for val in [1.0, -1.0]:
                    h[i] = val
                    new_val = _score(h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        improved = True
                        break
                    else:
                        h[i] = 0.0
        
        if not improved:
            break
    
    return h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = None
    best_val = 1e15

    # Phase 1: Try algebraic constructions
    candidates = []
    
    # Legendre sequence
    try:
        leg = _legendre_sequence(N)
        candidates.append(leg)
    except:
        pass
    
    # Barker-like
    candidates.append(_barker_like(N))
    
    # M-sequence-like (random with good autocorrelation)
    candidates.append(_m_sequence(N))
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Random binary sequences (multiple)
    rng = np.random.default_rng(42)
    for _ in range(50):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sparse sequences with various densities
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for _ in range(10):
            mask = rng.random(N) < density
            x = np.zeros(N)
            x[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            if abs(np.sum(x)) > 1e-6:
                candidates.append(x)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline - 30:
            break
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Phase 2: Local search on best candidate
    if time.time() < deadline - 10:
        x_ls, val_ls = _local_search_full_pass(best_x.copy(), max_passes=20, deadline=deadline - 10)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls
    
    # Phase 3: Sparse local search on best candidate
    if time.time() < deadline - 5:
        x_sp, val_sp = _local_search_sparse(best_x.copy(), max_passes=10, deadline=deadline - 5)
        if val_sp < best_val:
            best_val = val_sp
            best_x = x_sp
    
    # Phase 4: Try more algebraic initializations and local search
    if time.time() < deadline - 5:
        # Try shifted Legendre sequences
        for shift in range(1, 5):
            if time.time() > deadline - 5:
                break
            leg = _legendre_sequence(N)
            leg_shifted = np.roll(leg, shift)
            val = _score(leg_shifted)
            if val < best_val:
                best_val = val
                best_x = leg_shifted.copy()
                # Quick local search
                if time.time() < deadline - 3:
                    x_ls, val_ls = _local_search_full_pass(best_x.copy(), max_passes=10, deadline=deadline - 3)
                    if val_ls < best_val:
                        best_val = val_ls
                        best_x = x_ls
    
    # Phase 5: Final polish with sparse search on current best
    if time.time() < deadline - 2:
        x_final, val_final = _local_search_sparse(best_x.copy(), max_passes=5, deadline=deadline - 2)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final
    
    # Ensure valid result
    if best_x is None or abs(np.sum(best_x)) < 1e-6:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
