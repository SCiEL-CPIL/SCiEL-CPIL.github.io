import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))

def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6
    start = time.time()
    
    while time.time() - start < time_budget:
        # Choose block size: mostly small blocks, occasionally larger
        r = rng.random()
        if r < 0.5:
            L = int(rng.integers(1, 5))
        elif r < 0.8:
            L = int(rng.integers(4, 13))
        elif r < 0.95:
            L = int(rng.integers(8, 25))
        else:
            L = int(rng.integers(16, 33))
        L = min(L, N)
        
        i = int(rng.integers(0, N - L + 1))
        
        # Apply flip and compute new objective
        h[i:i+L] *= -1.0
        ac_new = _compute_acf(h)
        new_val = float(np.max(np.abs(ac_new[1:])))
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            h[i:i+L] *= -1.0
        
        T = max(T_min, T * 0.99995)
    
    return best_h, best

def _greedy_polish(h, time_budget, start):
    """Greedy single-flip polish: try each position, keep the best flip."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        best_ac = None
        
        for i in range(N):
            h[i] *= -1.0
            ac_try = _compute_acf(h)
            c = float(np.max(np.abs(ac_try[1:])))
            if c < best_c:
                best_c = c
                best_i = i
                best_ac = ac_try.copy()
            h[i] *= -1.0
        
        if best_i < 0:
            break
        
        h[best_i] *= -1.0
        ac = best_ac
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Long initial anneal from a random +-1 start.
        budget = min(40.0, hard_deadline - (time.time() - start) - 10.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts for diversity.
        restart_id = 1
        while time.time() - start < hard_deadline - 15.0:
            sub = min(15.0, hard_deadline - (time.time() - start) - 15.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start)
        if remaining > 2.0:
            h, c = _greedy_polish(global_best_h, remaining - 3.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)
