"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    where R[k] is the autocorrelation at lag k.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    C3 = max_sidelobe / R0
    return C3


def _init_autocorr(x):
    """Initialize autocorrelation array from x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    return R


def _get_c3_from_R(R):
    """Compute C3 from precomputed autocorrelation array."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    C3 = max_sidelobe / R0
    return C3


def _update_autocorr_flip(R, x, idx):
    """
    Update autocorrelation R after flipping x[idx].
    Returns new R and updated x.
    """
    R = R.copy()
    old_val = x[idx]
    x = x.copy()
    x[idx] = -old_val
    
    # For positive lags k = 1..N-1:
    for k in range(1, N):
        delta = 0.0
        if idx + k < N:
            delta -= 2.0 * old_val * x[idx + k]
        if idx - k >= 0:
            delta -= 2.0 * x[idx - k] * old_val
        R[k] += delta
    
    # For negative lags: R[N+k] corresponds to correlation at lag -k
    for k in range(1, N):
        delta = 0.0
        if idx >= k:
            delta -= 2.0 * old_val * x[idx - k]
        if idx + k < N:
            delta -= 2.0 * x[idx + k] * old_val
        R[N + k] += delta
    
    return R, x


def _thue_morse(n):
    """Generate Thue-Morse sequence: -1 if popcount(i) is odd, +1 otherwise."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


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


def _logistic_sequence(n, r=3.99, x0=0.5):
    """Generate a chaotic logistic map sequence and binarize it."""
    x = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x


def _block_pattern(n, block_size, phase=0):
    """Generate a block pattern: +1 for block_size, -1 for block_size, repeated."""
    seq = np.zeros(n)
    for i in range(n):
        pos = (i + phase) % (2 * block_size)
        seq[i] = 1.0 if pos < block_size else -1.0
    return seq


def _structured_initializations():
    """Generate diverse structured initializations."""
    inits = []
    
    # Thue-Morse (known to have good autocorrelation properties)
    inits.append(_thue_morse(N))
    
    # Van der Corput sequences
    inits.append(_van_der_corput(N, base=2))
    inits.append(_van_der_corput(N, base=3))
    inits.append(_van_der_corput(N, base=5))
    
    # Block patterns with various sizes
    for bs in [4, 8, 16, 32, 50]:
        inits.append(_block_pattern(N, bs, phase=0))
        inits.append(_block_pattern(N, bs, phase=1))
    
    # Logistic map sequences
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            inits.append(_logistic_sequence(N, r=r, x0=x0))
    
    # Mersenne Twister based (deterministic)
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    
    return inits


def _local_search(x, max_iter=50000, time_limit=None, multi_flip_prob=0.3):
    """
    Local search with incremental autocorrelation update for speed.
    Uses single flips and occasional multi-flip moves to escape local minima.
    """
    start_time = time.time()
    R = _init_autocorr(x)
    current_c3 = _get_c3_from_R(R)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 300
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Decide between single flip and multi-flip
        if random.random() < multi_flip_prob:
            # Multi-flip: flip 2-5 random positions
            num_flips = random.randint(2, 5)
            indices = random.sample(range(N), num_flips)
            x_trial = x.copy()
            R_trial = R.copy()
            for idx in indices:
                R_trial, x_trial = _update_autocorr_flip(R_trial, x_trial, idx)
        else:
            # Single flip
            idx = random.randint(0, N - 1)
            x_trial = x.copy()
            R_trial, x_trial = _update_autocorr_flip(R, x_trial, idx)
        
        new_c3 = _get_c3_from_R(R_trial)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_trial
            R = R_trial
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing(x, max_iter=50000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing with incremental autocorrelation update.
    """
    start_time = time.time()
    R = _init_autocorr(x)
    current_c3 = _get_c3_from_R(R)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        R_new, x_new = _update_autocorr_flip(R, x_new, idx)
        
        new_c3 = _get_c3_from_R(R_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _targeted_polish(x, R, time_limit=10.0):
    """
    Targeted polish: identify positions that contribute most to sidelobes
    and try flipping them.
    """
    start_time = time.time()
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = _get_c3_from_R(R)
    
    # Compute contribution of each position to the max sidelobe
    # For position i, its contribution to R[k] involves x[i]*x[i+k] and x[i-k]*x[i]
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try flipping a random position
        idx = random.randint(0, N - 1)
        x_trial = x.copy()
        R_trial, x_trial = _update_autocorr_flip(R, x_trial, idx)
        new_c3 = _get_c3_from_R(R_trial)
        
        if new_c3 < best_c3 - 1e-12:
            x = x_trial
            R = R_trial
            best_x = x.copy()
            best_R = R.copy()
            best_c3 = new_c3
    
    return best_x, best_R, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Structured initializations + local search
    structured_inits = _structured_initializations()
    for i, x_init in enumerate(structured_inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        # Give more time to earlier (likely better) initializations
        time_limit = max(8.0, 20.0 - i * 1.5)
        x_trial, c3_trial = _local_search(x_init.copy(), max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Random initializations + local search
    for trial in range(10):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=30000, time_limit=10)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _simulated_annealing(best_x, max_iter=50000, T_start=0.5, T_end=0.01,
                                         time_limit=remaining * 0.4)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Final targeted polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        R_final = _init_autocorr(best_x)
        x4, R4, c3_4 = _targeted_polish(best_x, R_final, time_limit=remaining * 0.8)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
