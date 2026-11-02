"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_fft(s):
    """Compute linear autoconvolution of sign sequence s via FFT."""
    n = 2 * N
    f = np.fft.rfft(s, n=n)
    ac = np.fft.irfft(f * np.conj(f), n=n)
    return ac[:N]

def _score(s):
    """Max absolute autoconvolution value for lags 1..N-1."""
    ac = _autoconv_fft(s)
    return np.max(np.abs(ac[1:]))

def _van_der_corput(N):
    """Van der Corput based initial signs."""
    signs = np.ones(N, dtype=np.float64)
    for i in range(N):
        x = 0.0
        d = 0.5
        n = i + 1
        while n > 0:
            x += d * (n % 2)
            n //= 2
            d /= 2
        signs[i] = 1.0 if x < 0.5 else -1.0
    return signs

def _greedy_descent(s, max_flips=5000, time_limit=8.0, batch_size=80):
    """Greedy coordinate descent to minimize max |AC[1:]|."""
    s = s.copy().astype(np.float64)
    start_time = time.time()
    current_val = _score(s)
    
    for iteration in range(max_flips):
        if time.time() - start_time > time_limit:
            break
        
        # Try a batch of random positions
        indices = np.random.choice(N, size=min(batch_size, N), replace=False)
        best_gain = 1e-10
        best_pos = -1
        best_new_val = current_val
        
        for idx in indices:
            s[idx] *= -1
            new_val = _score(s)
            s[idx] *= -1  # undo
            gain = current_val - new_val
            if gain > best_gain:
                best_gain = gain
                best_pos = idx
                best_new_val = new_val
        
        if best_pos >= 0:
            s[best_pos] *= -1
            current_val = best_new_val
        else:
            # Stuck, try a few more iterations
            if iteration > 100:
                break
    
    return s, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_s = None
    best_val = np.inf
    
    # Initializations to try
    inits = []
    inits.append(_van_der_corput(N))
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        rng = np.random.RandomState(seed)
        inits.append(rng.choice([1.0, -1.0], size=N))
    
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1.0
    inits.append(alt)
    
    # Modular
    mod = np.ones(N)
    for i in range(N):
        if (i * i) % 3 == 0:
            mod[i] = -1.0
    inits.append(mod)
    
    for r, s_init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        time_limit = min(6.0, remaining * 0.35)
        
        s_opt, val = _greedy_descent(s_init, max_flips=4000, time_limit=time_limit, batch_size=60)
        
        if val < best_val:
            best_val = val
            best_s = s_opt.copy()
        
        # If we've found a very good value, reduce time for remaining restarts
        if best_val < 0.005:
            time_limit = min(3.0, remaining * 0.25)
    
    if best_s is None:
        return _van_der_corput(N)
    
    return best_s
# EVOLVE-BLOCK-END
