"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0


def _compute_c3(x):
    """Compute C3 = max_{k!=0} |R[k]| / R[0] using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    # R[1:N] positive lags, R[N:2N-1] negative lags
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0


def _compute_R(x):
    """Compute full autocorrelation R[0..2N-2] via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)[:2 * N - 1]


def _c3_from_R(R):
    """Compute C3 from precomputed R. R has length 2N-1."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    # R[1:N] are lags 1..N-1, R[N:2N-1] are lags -(N-1)..-1
    # All non-zero lags
    sidelobes = R[1:]
    return np.max(np.abs(sidelobes)) / R0


def _flip_effect(x, i, R):
    """
    Compute new R after flipping x[i]. Vectorized.
    R_full has length 2N-1. R_full[k] for k=0..N-1 is autocorr at lag k.
    R_full[2N-1-k] for k=1..N-1 is autocorr at lag -k.
    Symmetry: R_full[k] = R_full[2N-1-k] for k=1..N-1.
    
    Flipping x[i] changes R_full[k] by:
    -2*x[i]*x[i+k] for k where i+k < N (term with j=i)
    -2*x[i-k]*x[i] for k where i-k >= 0 (term with j+k=i)
    """
    R_new = R.copy()
    xi = x[i]
    
    # Positive lags k=0..N-1
    # Term 1: x[i]*x[i+k] for k=0..N-1-i
    k1 = np.arange(0, N - i)
    R_new[k1] -= 2.0 * xi * x[i:i + len(k1)]
    
    # Term 2: x[i-k]*x[i] for k=0..i
    k2 = np.arange(0, i + 1)
    R_new[k2] -= 2.0 * x[i - k2][::-1] * xi
    
    # Apply symmetry: R_full[2N-1-k] = R_full[k] for k=1..N-1
    # Indices: for k in 1..N-1, R_full[2N-1-k] should equal R_full[k]
    # So R_full[N..2N-2] = R_full[N-1..1] reversed
    R_new[N:] = R_new[N-1:0:-1]
    
    return R_new


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with vectorized O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        if new_c3 < current_c3 - 1e-12:
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_R, best_c3


def _simulated_annealing_fast(x, R, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
    """Simulated annealing with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    T = T_start
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and random.random() < np.exp(-delta / T)):
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
    
    return best_x, best_R, best_c3


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


def _m_sequence(n, poly=0x5D, seed=1):
    """Generate m-sequence (maximal length LFSR)."""
    # Use a simple LFSR
    state = seed
    seq = np.zeros(n)
    for i in range(n):
        # Extract bit
        bit = state & 1
        seq[i] = 1.0 if bit == 0 else -1.0
        # Update state (simple Fibonacci LFSR)
        # For a 8-bit LFSR with polynomial x^8 + x^6 + x^5 + x^4 + 1
        # This is a simplified version; use numpy random as fallback
        pass
    # Fallback to random
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=n)


def _van_der_corput(n, base=2):
    """Van der Corput sequence mapped to +/-1."""
    seq = np.zeros(n)
    for i in range(n):
        v = 0.0
        f = 1.0 / base
        j = i + 1
        while j > 0:
            v += f * (j % base)
            j //= base
            f /= base
        seq[i] = 1.0 if v < 0.5 else -1.0
    return seq


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Initializations - more diverse
    inits = []
    inits.append(_thue_morse(N))
    inits.append(_van_der_corput(N))
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    rng2 = np.random.RandomState(67890)
    inits.append(rng2.choice([-1, 1], size=N))
    # Logistic map
    val = 0.5
    log_seq = np.zeros(N)
    for i in range(N):
        val = 3.99 * val * (1.0 - val)
        log_seq[i] = 1.0 if val > 0.5 else -1.0
    inits.append(log_seq)
    # Chaotic sequence with different parameter
    val2 = 0.7
    chaos_seq = np.zeros(N)
    for i in range(N):
        val2 = 3.9 * val2 * (1.0 - val2)
        chaos_seq[i] = 1.0 if val2 > 0.5 else -1.0
    inits.append(chaos_seq)
    
    # Local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(25.0, time_left * 0.25)
        if time_limit < 2.0:
            break
        x_trial, R_trial, c3_trial = _local_search_fast(x_init.copy(), R_init, max_iter=100000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(35.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(best_x.copy(), best_R.copy(), max_iter=200000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
