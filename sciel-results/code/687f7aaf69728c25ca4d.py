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
    """
    R_new = R.copy()
    xi = x[i]
    for k in range(N):
        delta = 0.0
        if i + k < N:
            delta -= 2.0 * xi * x[i + k]
        if i - k >= 0:
            delta -= 2.0 * x[i - k] * xi
        R_new[k] += delta
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


def _smooth_c3(x, beta=10.0):
    """
    Smooth surrogate for C3 using log-sum-exp on squared sidelobes.
    This provides a differentiable objective for gradient-based search.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    # Use log-sum-exp for smooth max approximation
    # max |R[k]| ≈ (1/beta) * log(sum exp(beta * |R[k]|))
    # But |R[k]| is not differentiable at 0, so use R[k]^2 for smoothness
    # max R[k]^2 ≈ (1/beta) * log(sum exp(beta * R[k]^2))
    # Then C3 ≈ sqrt(max R[k]^2) / R0
    R_sq = sidelobes * sidelobes
    max_val = np.max(R_sq)
    if max_val > 0:
        log_sum_exp = max_val / beta + np.log(np.sum(np.exp(beta * (R_sq - max_val))))
        smooth_max_sq = log_sum_exp / beta
    else:
        smooth_max_sq = 0.0
    return np.sqrt(smooth_max_sq) / R0


def _gradient_descent_smooth(x_init, max_iter=5000, beta=10.0, lr=0.01, time_limit=None):
    """
    Gradient descent on a smooth surrogate, with projection to {-1, +1}.
    Uses finite differences for gradient computation since analytic gradient
    of log-sum-exp of autocorrelation is complex.
    """
    start_time = time.time()
    x = x_init.copy()
    best_x = x.copy()
    best_c3 = _compute_c3(x)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(N)
        eps = 0.1
        f0 = _smooth_c3(x, beta)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] = -x_plus[i]  # flip
            f_plus = _smooth_c3(x_plus, beta)
            grad[i] = (f_plus - f0) / (2.0 * x[i])  # approximate gradient
        
        # Gradient step
        x = x - lr * grad
        
        # Project to nearest {-1, +1}
        x = np.sign(x)
        x[x == 0] = 1.0
        
        # Evaluate actual C3
        c3 = _compute_c3(x)
        if c3 < best_c3:
            best_x = x.copy()
            best_c3 = c3
    
    return best_x, best_c3


def _construct():
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
    inits.append(_logistic_sequence(N, r=3.99, x0=0.5))
    inits.append(_logistic_sequence(N, r=3.9, x0=0.3))
    inits.append(_van_der_corput(N, base=2))
    inits.append(_van_der_corput(N, base=3))
    
    # Local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(20.0, time_left * 0.25)
        if time_limit < 2.0:
            break
        x_trial, R_trial, c3_trial = _local_search_fast(x_init.copy(), R_init, max_iter=30000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Smooth gradient descent from best
    if time.time() < DEADLINE - TIME_MARGIN - 10.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.4)
        if time_limit > 5.0:
            x_gd, c3_gd = _gradient_descent_smooth(best_x.copy(), max_iter=3000, beta=10.0, lr=0.05, time_limit=time_limit)
            if c3_gd < best_c3:
                best_x = x_gd
                best_R = _compute_R(best_x)
                best_c3 = c3_gd
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(25.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=40000, time_limit=time_limit)
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
