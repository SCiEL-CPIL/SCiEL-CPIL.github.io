import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin from 150s

def _autoconv_linear(h):
    """Compute linear autoconvolution of h."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _max_abs_ac_offpeak(ac):
    """Max absolute autoconvolution excluding k=0 (and symmetric)."""
    # ac has length 2N-1, index N-1 is k=0
    # We exclude the peak at k=0
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    return np.max(np.abs(ac_no_peak))

def _score(h):
    """Compute the ratio: max |C(k)| for k!=0 divided by (sum h)^2."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return np.inf
    ac = _autoconv_linear(h)
    max_abs = _max_abs_ac_offpeak(ac)
    return max_abs / (S * S)

def _soft_max_abs(ac, beta):
    """Smooth approximation of max |ac| excluding peak."""
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    abs_ac = np.abs(ac_no_peak)
    if beta < 1e6:
        return np.logaddexp.reduce(np.log(abs_ac + 1e-30) * beta) / beta
    else:
        return np.max(abs_ac)

def _smooth_obj(h, beta):
    """Smooth objective for gradient-based optimization."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return 1e10
    ac = _autoconv_linear(h)
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (S * S)

def _local_search_flip(h, max_iters=5000):
    """Sign-flip local search to minimize max |C(k)| for k!=0."""
    h = h.copy().astype(float)
    # Normalize so sum = 1
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    
    best_score = _score(h)
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            h[i] = -h[i]
            S = np.sum(h)
            if abs(S) < 1e-15:
                h[i] = -h[i]
                continue
            ac = _autoconv_linear(h)
            max_abs = _max_abs_ac_offpeak(ac)
            new_score = max_abs / (S * S)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h

def _local_search_continuous(h, max_iters=200, lr=0.01):
    """Simple gradient-free local search on continuous values."""
    h = h.copy().astype(float)
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    best_score = _score(h)
    
    for it in range(max_iters):
        # Try small perturbations
        improved = False
        for _ in range(20):
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.1, 0.1)
            h[idx] += delta
            S = np.sum(h)
            if abs(S) < 1e-15:
                h[idx] -= delta
                continue
            ac = _autoconv_linear(h)
            max_abs = _max_abs_ac_offpeak(ac)
            new_score = max_abs / (S * S)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[idx] -= delta
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try several random initializations
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Different initialization strategies
        if restart == 0:
            h = np.ones(N)
        elif restart == 1:
            # Chirp-like
            t = np.arange(N)
            h = np.cos(2 * np.pi * t * t / N)
        elif restart == 2:
            # Random signs
            h = np.random.choice([-1.0, 1.0], size=N)
            if abs(np.sum(h)) < 1e-10:
                h[0] = 1.0
        elif restart == 3:
            # Random smooth
            h = np.random.randn(N)
            h = np.convolve(h, np.ones(5)/5, mode='same')
        elif restart == 4:
            # Barker-like (all ones then flip some)
            h = np.ones(N)
            for i in range(0, N, 4):
                h[i] = -1
        elif restart == 5:
            # Zadoff-Chu like
            t = np.arange(N)
            h = np.real(np.exp(1j * np.pi * t * (t + 1) / N))
        elif restart == 6:
            # Flat top window
            h = np.ones(N)
            h[:N//4] = 0.5
            h[3*N//4:] = 0.5
        elif restart == 7:
            # Random with bias
            h = np.random.uniform(-1, 1, size=N)
            h += 0.5  # bias positive
        elif restart == 8:
            # Sine wave
            t = np.arange(N)
            h = np.sin(2 * np.pi * t / N)
            h += 0.5
        elif restart == 9:
            # Two-segment
            h = np.ones(N)
            h[N//2:] = -0.5
        else:
            # More random
            h = np.random.randn(N)
            h += np.random.uniform(0.5, 2.0)
        
        # Normalize
        S = np.sum(h)
        if abs(S) < 1e-15:
            h[0] += 1.0
            S = np.sum(h)
        h = h / S
        
        # Continuous local search
        if time.time() - start_time > DEADLINE - 5:
            break
        h = _local_search_continuous(h, max_iters=100)
        
        # Sign-flip local search
        if time.time() - start_time > DEADLINE - 5:
            break
        h = _local_search_flip(h, max_iters=3000)
        
        # Evaluate
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    return best_h
