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
    """Compute C3 from precomputed R."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _flip_effect(x, i, R):
    """
    Compute new R after flipping x[i].
    Flipping x[i] -> -x[i] changes R[k] by:
      For lag k (where 0 <= k < N): R[k] += -2*x[i]*x[i+k] for k=0..N-1-i
      Also for the symmetric part.
    Actually, R_full[k] = sum_j x[j]*x[j+k] for k>=0, and R_full[2N-2-k] = R_full[k].
    Flipping x[i]: the terms involving x[i] are x[i]*x[i+k] for k=0..N-1-i (positive lags)
    and x[i-k]*x[i] for k=0..i (which correspond to lags where i-k >= 0, i.e., lag = k from position i-k to i).
    
    More carefully: R[k] = sum_{j=0}^{N-1-k} x[j] x[j+k]
    After flip at i: x'[i] = -x[i]
    R'[k] = R[k] - 2*x[i]*x[i+k] if i+k < N (i.e., k <= N-1-i)
           R'[k] = R[k] - 2*x[i-k]*x[i] if i-k >= 0 and k <= i (i.e., the term x[i-k]*x[i] in R[k])
    
    Wait, R[k] = sum_{j=0}^{N-1-k} x[j] x[j+k]. The term involving x[i] appears when:
    - j = i: requires i <= N-1-k, i.e., k <= N-1-i. Term: x[i]*x[i+k]
    - j+k = i: requires j = i-k >= 0, i.e., k <= i. Term: x[i-k]*x[i]
    
    These are the same term only when i = i-k... no. They are different terms unless k=0.
    For k > 0: if both conditions hold (k <= N-1-i AND k <= i), then both terms x[i]*x[i+k] and x[i-k]*x[i] are present in R[k].
    
    So delta_R[k] = -2*x[i]*x[i+k] (if k <= N-1-i) - 2*x[i-k]*x[i] (if k <= i)
    
    For the full R array of length 2N-1:
    R_full[k] for k in 0..N-1 is the positive lag autocorrelation.
    R_full[N+k] = R_full[N-1-k] for k in 0..N-2 (symmetric).
    
    Actually from FFT: R_full has length 2N-1, R_full[0] = sum x[i]^2, R_full[k] for k=1..N-1 is sum x[i]x[i+k], and R_full[N+k] = R_full[N-1-k].
    """
    R_new = R.copy()
    xi = x[i]
    # For each lag k in 0..N-1:
    # Term 1: j=i, contributes x[i]*x[i+k] if i+k < N
    # Term 2: j+k=i, contributes x[i-k]*x[i] if i-k >= 0
    for k in range(N):
        delta = 0.0
        if i + k < N:
            delta -= 2.0 * xi * x[i + k]
        if i - k >= 0:
            delta -= 2.0 * x[i - k] * xi
        R_new[k] += delta
        # Symmetric part: R_full[2N-2-k] corresponds to negative lag k
        # R_full[N+k] = R_full[N-1-k]... let me just recompute symmetry
    # Reimpose symmetry: R_full[2N-2-k] = R_full[k] for k=0..N-1
    # Actually R_full is already symmetric from FFT. After modification, we need to fix.
    # R_full[k] and R_full[2N-2-k] should be equal.
    for k in range(N):
        R_new[2*N-2-k] = R_new[k]
    return R_new


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 200
    
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


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Initializations
    inits = []
    inits.append(_thue_morse(N))
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
    
    # Local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.3)
        if time_limit < 2.0:
            break
        x_trial, R_trial, c3_trial = _local_search_fast(x_init.copy(), R_init, max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(40.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
