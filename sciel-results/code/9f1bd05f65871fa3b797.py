"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

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
    """Compute new R after flipping x[i] using O(N) update."""
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


def _continuous_optimize(x0, time_limit=30.0, start_time=None):
    """
    Use L-BFGS-B to optimize a continuous surrogate, then project to discrete.
    Objective: soft-max of |conv| / (sum|h|)^2
    """
    if start_time is None:
        start_time = time.time()
    
    def objective(z):
        # Check deadline
        if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
            return 1e10
        
        h = np.tanh(z)
        # Compute autoconvolution via FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        conv = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
        # Smooth absolute value and soft-max
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        T = 1.0
        max_abs = np.max(abs_conv)
        shifted = (abs_conv - max_abs) / T
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        value = soft_max / (sum_abs_h ** 2)
        return value
    
    def gradient(z):
        # Numerical gradient (expensive but safe)
        # Actually, let's use analytical gradient approximation
        # For now, use finite differences on a subset
        h = np.tanh(z)
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        conv = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        T = 1.0
        max_abs = np.max(abs_conv)
        shifted = (abs_conv - max_abs) / T
        shifted = np.clip(shifted, -100, 100)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        soft_max = T * (np.log(np.sum(np.exp(shifted))) + max_abs / T)
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        # d(soft_max)/d(conv[k]) = weights[k] * conv[k] / abs_conv[k] (for smooth abs)
        d_conv = weights * conv / np.maximum(abs_conv, eps)
        
        # d(conv)/d(h) = 2 * convolve(h, delta) = 2*h reversed for autocorrelation
        # For autocorrelation R[k] = sum_j h[j]h[j+k], dR/dh[i] = sum_k (dR[k]/dh[i])
        # This is complex; use numerical gradient
        grad = np.zeros_like(z)
        step = 1e-5
        for i in range(0, N, max(1, N // 50)):  # Sparse gradient for speed
            z_plus = z.copy()
            z_plus[i] += step
            z_minus = z.copy()
            z_minus[i] -= step
            grad[i] = (objective(z_plus) - objective(z_minus)) / (2 * step)
        
        # Interpolate gradient
        if N > 50:
            indices = np.arange(0, N, max(1, N // 50))
            if len(indices) > 1:
                grad_full = np.interp(np.arange(N), indices, grad[::max(1, N//50)])
            else:
                grad_full = grad
        else:
            grad_full = grad
        
        # Chain rule: d/dz = d/dh * dh/dz = d/dh * (1 - h^2)
        dh_dz = 1 - h**2
        return grad_full * dh_dz
    
    # Run L-BFGS-B
    result = minimize(
        objective, x0, method='L-BFGS-B', jac=None,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
    )
    
    if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
        return None
    
    h_cont = np.tanh(result.x)
    # Project to discrete ±1
    h_disc = np.sign(h_cont)
    h_disc[h_disc == 0] = 1.0
    return h_disc


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Phase 1: Continuous optimization to find good basins
    # Try several initializations in continuous space
    continuous_inits = []
    
    # All ones (seed)
    continuous_inits.append(np.full(N, 3.0))
    
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 2.0
    alt[1::2] = -2.0
    continuous_inits.append(alt)
    
    # Random continuous
    rng = np.random.RandomState(999)
    continuous_inits.append(rng.randn(N) * 1.5)
    
    # Thue-Morse mapped to continuous
    tm = _thue_morse(N)
    continuous_inits.append(tm * 2.0)
    
    # Blocky random
    block_size = 10
    n_blocks = N // block_size
    block_vals = np.random.RandomState(555).choice([-1, 1], size=n_blocks) * 2.0
    continuous_inits.append(np.repeat(block_vals, block_size))
    
    discrete_candidates = [best_x.copy()]
    
    for i, z_init in enumerate(continuous_inits):
        if time.time() > DEADLINE - TIME_MARGIN - 5.0:
            break
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 5.0:
            break
        time_limit = min(25.0, time_left * 0.4)
        if time_limit < 3.0:
            break
        
        h_disc = _continuous_optimize(z_init.copy(), time_limit=time_limit, start_time=start_time)
        if h_disc is not None:
            discrete_candidates.append(h_disc)
    
    # Phase 2: Discrete local search from each candidate
    for i, x_init in enumerate(discrete_candidates):
        if time.time() > DEADLINE - TIME_MARGIN - 3.0:
            break
        R_init = _compute_R(x_init)
        c3_init = _c3_from_R(R_init)
        
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 3.0:
            break
        time_limit = min(20.0, time_left * 0.3)
        if time_limit < 2.0:
            break
        
        x_trial, R_trial, c3_trial = _local_search_fast(
            x_init.copy(), R_init, max_iter=50000, time_limit=time_limit
        )
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Phase 3: Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(35.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(
                best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit
            )
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Phase 4: Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(
            best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit
        )
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
