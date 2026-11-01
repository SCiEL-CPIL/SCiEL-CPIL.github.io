"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 10.0


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    
    For a signed step function with values in {-1, +1}, the autocorrelation is:
      R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]  for k = 0, ..., N-1
    
    The "upper bound on C3" is the maximum absolute sidelobe relative to main lobe:
      C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    # Maximum absolute sidelobe (excluding main lobe)
    max_sidelobe = np.max(np.abs(R[1:]))
    
    C3 = max_sidelobe / R0
    return C3


def _eval_flip(s, acf, i):
    """Compute peak sidelobe after flipping position i without full FFT."""
    si = s[i]
    new_acf = acf.copy()
    # For k where i+k < N: new_acf[k] -= 2*si*s[i+k]
    k_max = N - i
    if k_max > 0:
        new_acf[:k_max] -= 2 * si * s[i:i+k_max]
    # For k where i-k >= 0, i.e., k <= i: new_acf[k] -= 2*s[k]*si
    if i > 0:
        new_acf[:i+1] -= 2 * s[:i+1] * si
    return np.max(np.abs(new_acf[1:]))


def _local_search(s, deadline):
    """
    Iteratively flip the best single position until no improvement or deadline.
    Returns (best_seq, best_val).
    """
    acf = _compute_acf(s)
    current_val = np.max(np.abs(acf[1:]))
    best_seq = s.copy()
    best_val = current_val
    
    improved = True
    while improved and time.time() < deadline - TIME_MARGIN:
        improved = False
        best_flip_val = current_val
        best_flip_idx = -1
        
        for i in range(N):
            new_val = _eval_flip(s, acf, i)
            if new_val < best_flip_val:
                best_flip_val = new_val
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            s[best_flip_idx] = -s[best_flip_idx]
            acf = _compute_acf(s)
            current_val = np.max(np.abs(acf[1:]))
            improved = True
            if current_val < best_val:
                best_val = current_val
                best_seq = s.copy()
    
    return best_seq, best_val


def _compute_acf(s):
    """Compute autocorrelation using FFT."""
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf


def _logistic_sequence(n, r=3.99, x0=0.5):
    """Generate a chaotic logistic map sequence and binarize it."""
    x = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x


def _van_der_corput(n, base=2):
    """Generate a van der Corput low-discrepancy sequence and binarize it."""
    seq = np.zeros(n)
    for i in range(1, n + 1):
        v = 0.0
        f = 1.0 / base
        m = i
        while m > 0:
            m, r = divmod(m, base)
            v += r * f
            f /= base
        seq[i-1] = 1.0 if v > 0.5 else -1.0
    return seq


def _thue_morse(n):
    """Generate a Thue-Morse sequence."""
    tm = np.zeros(n)
    for i in range(n):
        c = 0
        m = i
        while m:
            c += m & 1
            m >>= 1
        tm[i] = 1.0 if c % 2 == 0 else -1.0
    return tm


def _chaos_initializations():
    """Generate multiple chaotic/quasi-random initializations."""
    inits = []
    
    # Logistic map with different parameters
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            inits.append(_logistic_sequence(N, r=r, x0=x0))
    
    # Van der Corput sequences
    inits.append(_van_der_corput(N, base=2))
    inits.append(_van_der_corput(N, base=3))
    inits.append(_van_der_corput(N, base=5))
    
    # Thue-Morse-like
    inits.append(_thue_morse(N))
    
    # Mersenne Twister based (deterministic)
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    
    return inits


def construct():
    """
    Main construction function that searches for a good signed step function.
    Returns a sequence of signed step heights.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    random.seed(42)
    np.random.seed(42)
    
    best_seq = None
    best_val = np.inf
    
    # Strategy 1: Chaotic/quasi-random initializations + local search
    chaos_inits = _chaos_initializations()
    for i, x_init in enumerate(chaos_inits):
        if time.time() > deadline - TIME_MARGIN:
            break
        x_trial, c3_trial = _local_search(x_init.copy(), deadline)
        if c3_trial < best_val:
            best_val = c3_trial
            best_seq = x_trial.copy()
    
    # Strategy 2: Standard random initialization + local search
    for trial in range(10):
        if time.time() > deadline - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x_trial, c3_trial = _local_search(x_init.copy(), deadline)
        if c3_trial < best_val:
            best_val = c3_trial
            best_seq = x_trial.copy()
    
    if best_seq is None:
        return np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
