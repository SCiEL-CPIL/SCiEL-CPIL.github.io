"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _autoconv_bound(f):
    """Compute max_t |f*f(t)| for a step function f on [-1/4,1/4] with N intervals."""
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))

def _candidate_m_sequence():
    """Generate an m-sequence of length N."""
    degree = 9  # 2^9 - 1 = 511 >= 400
    state = 1
    seq = []
    for _ in range(2**degree - 1):
        seq.append(1.0 if (state & 1) else -1.0)
        feedback = (state & 1) ^ ((state >> 4) & 1)
        state = (state >> 1) | (feedback << (degree - 1))
    return np.array(seq[:N])

def _candidate_legendre():
    """Legendre sequence based on quadratic residues mod p."""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    p = N
    while not is_prime(p):
        p += 1
    seq = np.zeros(N)
    for i in range(N):
        if i == 0:
            seq[i] = 1.0
        else:
            if pow(i, (p - 1) // 2, p) == 1:
                seq[i] = 1.0
            else:
                seq[i] = -1.0
    return seq

def _candidate_smooth_waveform(seed=0, max_freq=20, n_harmonics=5):
    """
    Generate a smooth oscillating candidate by summing low-frequency sinusoids.
    This creates a continuous waveform sampled at N points, then discretize to signed steps.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Generate random low-frequency sinusoidal components
    waveform = np.zeros(N)
    for k in range(1, max_freq + 1):
        # Only include some harmonics for smoothness
        if rng.random() < 0.5:  # randomly skip some frequencies
            amp = rng.uniform(0.1, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            waveform += amp * np.sin(2 * np.pi * k * t + phase)
    
    # Normalize and discretize to signed step function
    # Use a threshold to create a signed step function
    threshold = 0.0
    f = np.sign(waveform)
    # Ensure we have both +1 and -1
    if np.all(f == 1):
        f[::2] = -1.0
    elif np.all(f == -1):
        f[::2] = 1.0
    return f

def _local_search(f, max_sweeps=200):
    """Greedy local search: flip bits to reduce max autoconvolution."""
    f = f.copy()
    best_val = _autoconv_bound(f)
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE:
                break
            f[i] = -f[i]
            val = _autoconv_bound(f)
            if val < best_val:
                best_val = val
                improved = True
            else:
                f[i] = -f[i]
        if not improved:
            break
    return f, best_val

def construct():
    # Start with the original discrete candidates
    candidates = [
        _candidate_m_sequence(),
        _candidate_legendre(),
        np.where(np.arange(N) % 2 == 0, 1.0, -1.0),
    ]
    
    # Add smooth waveform candidates
    for seed in range(8):
        if time.time() > _DEADLINE:
            break
        candidates.append(_candidate_smooth_waveform(seed=seed, max_freq=15, n_harmonics=5))
    
    # Add random discrete candidates
    rng = np.random.RandomState(42)
    for _ in range(3):
        if time.time() > _DEADLINE:
            break
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    best = None
    best_val = float('inf')
    
    for c in candidates:
        if time.time() > _DEADLINE:
            break
        c_opt, val = _local_search(c)
        if val < best_val:
            best_val = val
            best = c_opt
    
    if best is None:
        return np.ones(N)
    return best
# EVOLVE-BLOCK-END
