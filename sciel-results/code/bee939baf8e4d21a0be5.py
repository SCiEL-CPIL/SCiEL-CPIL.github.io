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

def _generate_msequence(L):
    """Generate an m-sequence of length L = 2^m - 1 using a primitive polynomial."""
    # Find a primitive polynomial for the given m
    # Try common primitive polynomials
    polys = {
        1: [0],  # trivial
        2: [1, 0],  # x^2 + 1
        3: [1, 0, 1],  # x^3 + x + 1
        4: [1, 0, 0, 1],  # x^4 + x + 1
        5: [1, 0, 0, 0, 1],  # x^5 + x^2 + 1
        6: [1, 0, 0, 1, 1, 1],  # x^6 + x^4 + x^3 + x^2 + 1
        7: [1, 0, 0, 0, 0, 1, 1],  # x^7 + x + 1
        8: [1, 0, 0, 1, 0, 0, 0, 1],  # x^8 + x^4 + x^3 + x^2 + 1
        9: [1, 0, 0, 0, 0, 1, 1, 0, 1],  # x^9 + x^5 + x^4 + x^2 + 1
        10: [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],  # x^10 + x^7 + x^6 + x^4 + x^3 + x^2 + 1
        11: [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # x^11 + x^7 + x^6 + x^4 + x^3 + x^2 + 1
        12: [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],  # x^12 + x^9 + x^6 + x^4 + x + 1
        13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # x^13 + x^2 + 1
    }
    
    # Find m such that 2^m - 1 >= N, then truncate
    m = 1
    while (1 << m) - 1 < N:
        m += 1
    
    L_full = (1 << m) - 1
    poly = polys.get(m, None)
    if poly is None:
        # Fallback: use a simple LFSR
        # Use x^m + x^(m-1) + ... as fallback
        poly = [1] + [0] * (m - 1) + [1]
    
    # Generate LFSR sequence
    # State is m bits, start with all 1s
    state = [1] * m
    seq = []
    for _ in range(L_full):
        seq.append(1 if state[-1] else -1)
        # Feedback: XOR of bits where poly has 1
        fb = 0
        for i, p in enumerate(poly):
            if p:
                fb ^= state[i]
        # Shift right, insert feedback at left
        state = [fb] + state[:-1]
    
    return np.array(seq[:N], dtype=np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Precompute phase factors for incremental FFT updates
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    def _score_from_fft(X, s):
        """Compute score from FFT and sum."""
        if abs(s) < 1e-12:
            return float('inf')
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2 * N - 1]
        return np.max(np.abs(ac)) / (s * s)

    # Phase 1: Try structured candidates
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # M-sequences (have nearly flat autocorrelation)
    try:
        mseq = _generate_msequence(N)
        sc = _score(mseq)
        if sc < best_score:
            best_score = sc
            best_x = mseq.copy()
    except Exception:
        pass

    # Other structured candidates
    candidates = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
        np.ones(N//2) + np.ones(N - N//2) * (-1.0),
    ]
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Random ±1 candidates
    rng = np.random.default_rng(42)
    for seed in range(200):
        if time.time() > deadline - 80:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Phase 2: Fast local search with incremental FFT
    x = best_x.copy()
    cur_score = best_score
    X = np.fft.rfft(x, fft_size)
    cur_sum = np.sum(x)

    rng2 = np.random.default_rng(999)
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline - 20 and no_improve < max_no_improve:
        idx = rng2.integers(0, N)
        old_val = x[idx]
        new_val = -old_val
        delta = new_val - old_val  # -2 * old_val

        # Update FFT incrementally
        X_new = X + delta * exp_factors[idx]
        new_sum = cur_sum + delta

        new_score = _score_from_fft(X_new, new_sum)

        if new_score < cur_score:
            x[idx] = new_val
            X = X_new
            cur_sum = new_sum
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            no_improve += 1

    # Phase 3: Systematic sweep for final refinement
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
