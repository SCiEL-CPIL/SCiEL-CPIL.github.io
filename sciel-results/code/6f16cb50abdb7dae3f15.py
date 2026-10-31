# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_abs_autoconv(x):
    """Compute max |autoconvolution| for a real sequence x (circular)."""
    n = len(x)
    # Circular autoconvolution via FFT
    fx = np.fft.rfft(x, n=2 * n)
    autoc = np.fft.irfft(fx * np.conj(fx), n=2 * n)
    autoc = autoc[:n]  # take first n lags (circular)
    return np.max(np.abs(autoc))


def _score(x):
    """Score = max |autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _max_abs_autoconv(x) / (s * s)


def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n (+1/-1)."""
    t = np.zeros(n)
    for i in range(n):
        # Count bits
        c = 0
        tmp = i
        while tmp:
            c += tmp & 1
            tmp >>= 1
        t[i] = 1.0 if c % 2 == 0 else -1.0
    return t


def _local_sign_flip_optimize(x, max_iters=5000):
    """Optimize signs by flipping individual elements to reduce score."""
    current_score = _score(x)
    if np.isinf(current_score):
        return x
    best_x = x.copy()
    n = len(x)
    # Precompute autoconv for current x
    # To speed up, we can update incrementally, but for N=400, FFT is fast enough for a few hundred iterations.
    # However, 5000 iterations * FFT(800) might be slow.
    # Let's limit iterations or use a smarter check.
    # Actually, just a few hundred iterations of random flips or greedy flips.
    
    # Greedy local search: try flipping each index, if score improves, keep it.
    # This is O(N * cost_score). cost_score is O(N log N). Total O(N^2 log N).
    # N=400, N^2=160k, log N ~ 9. 1.4M ops per pass.
    # 10 passes = 14M ops. Should be fast in NumPy?
    # Let's try 20 passes.
    
    for _ in range(20):
        improved = False
        for i in range(n):
            old_val = best_x[i]
            best_x[i] = -old_val
            new_score = _score(best_x)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                best_x[i] = old_val
        if not improved:
            break
    return best_x


def construct():
    """
    Construct signed step heights minimizing max|autoconv| / (integral)^2.
    Tries square wave family and Thue-Morse based family.
    """
    deadline = time.time() + 110.0  # Safety margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # 1. Square Wave Family (Directive Strategy)
    # x = [1]*k + [-1]*(N-k)
    # Note: k=N is all ones. k=0 is all minus.
    for k in range(1, N):
        if time.time() > deadline:
            break
        x_sq = np.empty(N)
        x_sq[:k] = 1.0
        x_sq[k:] = -1.0
        sc = _score(x_sq)
        if sc < best_score:
            best_score = sc
            best_x = x_sq.copy()
            
    # 2. Thue-Morse Family (Inspiration Strategy)
    if time.time() < deadline - 10.0:
        x_tm = _thue_morse_sequence(N)
        # Initial score
        sc_tm = _score(x_tm)
        if sc_tm < best_score:
            best_score = sc_tm
            best_x = x_tm.copy()
            
        # Local optimization
        if time.time() < deadline - 5.0:
            x_opt = _local_sign_flip_optimize(best_x, max_iters=20)
            sc_opt = _score(x_opt)
            if sc_opt < best_score:
                best_score = sc_opt
                best_x = x_opt

    return best_x
# EVOLVE-BLOCK-END
