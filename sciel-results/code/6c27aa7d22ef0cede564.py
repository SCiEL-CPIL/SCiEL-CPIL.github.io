"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 150.0
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_autocorr(x):
    """Compute linear autocorrelation of x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    return R


def _max_sidelobe(R):
    """Return maximum absolute sidelobe value from autocorrelation R."""
    R0 = R[0]
    if R0 == 0:
        return 0.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _soft_max_sidelobe(R, beta=20.0):
    """
    Soft-max approximation of max|sidelobes|/R0 using log-sum-exp.
    Returns (value, gradient_wrt_R)
    """
    R0 = R[0]
    if R0 == 0:
        return 1.0, np.zeros_like(R)
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    # Soft max: (1/beta) * log(sum(exp(beta * x_i)))
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-15:
        return 0.0, np.zeros_like(R)
    
    scaled = beta * abs_sidelobes
    scaled_shifted = scaled - np.max(scaled)
    exp_scaled = np.exp(scaled_shifted)
    log_sum = np.log(np.sum(exp_scaled)) + np.max(scaled) / beta
    
    soft_max_val = log_sum / beta / R0
    
    # Gradient of soft-max w.r.t. abs_sidelobes
    weights = exp_scaled / np.sum(exp_scaled)
    grad_abs = weights / beta / R0
    
    # Chain rule: d|s|/ds = sign(s)
    grad_sidelobes = grad_abs * np.sign(sidelobes)
    
    # Map back to R
    grad_R = np.zeros_like(R)
    grad_R[1:N] = grad_sidelobes[:N-1]
    grad_R[N:2*N-1] = grad_sidelobes[N-1:]
    
    # Also need gradient w.r.t. R0 (since we divide by R0)
    # d(soft_max/R0)/dR0 = -soft_max/R0^2 * ... but this is complex
    # For simplicity, we'll handle R0 in the objective differently
    
    return soft_max_val, grad_R


def _objective_and_gradient(x, beta=20.0):
    """
    Objective: soft-max of |sidelobes| / R0
    x is a continuous vector (not necessarily ±1)
    """
    # Ensure x is real
    x = np.real(x)
    
    R = _compute_autocorr(x)
    R0 = R[0]
    
    if R0 < 1e-15:
        return 1.0, np.zeros_like(x)
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft max
    scaled = beta * abs_sidelobes
    scaled_shifted = scaled - np.max(scaled)
    exp_scaled = np.exp(scaled_shifted)
    sum_exp = np.sum(exp_scaled)
    log_sum = np.log(sum_exp) + np.max(scaled) / beta
    soft_max_val = log_sum / beta / R0
    
    # Gradient of soft_max w.r.t. sidelobes
    weights = exp_scaled / sum_exp
    grad_sidelobes = weights * beta / beta / R0  # = weights / R0
    grad_sidelobes = grad_sidelobes * np.sign(sidelobes)
    
    # Gradient w.r.t. R0: d(soft_max/R0)/dR0 = -soft_max/R0^2
    # But soft_max itself depends on R0 only through the division
    # Actually, soft_max = log_sum_exp / beta / R0
    # d(soft_max)/dR0 = -log_sum_exp / (beta * R0^2)
    # = -soft_max / R0
    
    # Full gradient w.r.t. R:
    grad_R = np.zeros_like(R)
    grad_R[1:N] = grad_sidelobes[:N-1]
    grad_R[N:2*N-1] = grad_sidelobes[N-1:]
    grad_R[0] = -soft_max / R0  # contribution from 1/R0 factor
    
    # Now gradient w.r.t. x via autocorrelation
    # R[k] = sum_i x[i] * x[i+k] for k >= 0
    # dR[k]/dx[j] = x[j+k] if j+k < N, plus x[j-k] if j >= k
    # This is equivalent to: grad_x = IFFT(|FFT(x)|^2 gradient)
    # More precisely, use the FFT relationship:
    # R = IFFT(|FFT(x)|^2)
    # dR/dx can be computed via: grad_x = 2 * IFFT(FFT(x) * conj(FFT(grad_R)))
    
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    GR = np.fft.rfft(grad_R, n=n_fft)
    
    # dR/dx: R = IFFT(F * conj(F)), so dR = IFFT(F * conj(dF) + conj(F) * dF)
    # dF = FFT(dx), so dR/dx involves F * conj(F) structure
    # Actually, for R = IFFT(|F|^2), the gradient is:
    # grad_x = 2 * real(IFTT(F * conj(FFT(grad_R))))
    
    grad_x = 2.0 * np.fft.irfft(F * np.conj(GR), n=n_fft)[:N]
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad_x)
    if grad_norm > 1e-15:
        grad_x = grad_x / grad_norm * grad_norm  # keep as is
    
    return soft_max_val, grad_x


def _local_search_discrete(x, max_iter=5000, time_limit=None):
    """
    Discrete local search: flip individual bits and accept if C3 improves.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_c3 = _max_sidelobe(_compute_autocorr(x))
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _max_sidelobe(_compute_autocorr(x_new))
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
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


def _simulated_annealing(x_init, time_limit=30.0, max_iter=20000, t_start=1.0, t_end=0.01):
    """
    Simulated Annealing for binary sequences.
    Uses geometric cooling schedule.
    """
    start_time = time.time()
    x = np.sign(x_init).astype(float)
    current_c3 = _max_sidelobe(_compute_autocorr(x))
    best_x = x.copy()
    best_c3 = current_c3
    
    # Precompute log cooling factor
    log_cooling = np.log(t_end / t_start) / max_iter if max_iter > 0 else 0
    
    t = t_start
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Flip a random bit
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _max_sidelobe(_compute_autocorr(x_new))
        
        delta = new_c3 - current_c3
        
        if delta < 0:
            # Accept improvement
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            # Accept with probability exp(-delta / t)
            if random.random() < np.exp(-delta / max(t, 1e-10)):
                x = x_new
                current_c3 = new_c3
        
        # Cool down
        t *= np.exp(log_cooling)
    
    return best_x, best_c3


def _gradient_based_search(x_init, beta=20.0, maxiter=30, maxfun=100, time_limit=None):
    """
    Gradient-based optimization using L-BFGS-B on the continuous relaxation.
    """
    start_time = time.time()
    x = x_init.astype(float).copy()
    
    def obj_func(v):
        val, grad = _objective_and_gradient(v, beta=beta)
        return val, grad
    
    # Use L-BFGS-B
    result = minimize(
        obj_func,
        x,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    # Round to ±1
    x_rounded = np.sign(result.x)
    x_rounded[x_rounded == 0] = 1.0
    c3_rounded = _max_sidelobe(_compute_autocorr(x_rounded))
    
    return x_rounded, c3_rounded


def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n as ±1 values."""
    tm = [0]
    for i in range(1, n):
        tm.append(tm[i // 2] ^ (i % 2))
    return np.array(tm[:n], dtype=float)


def _construct():
    """
    Main construction function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _max_sidelobe(_compute_autocorr(best_x))
    
    # Strategy 1: Start with all ones, do discrete local search
    x1, c3_1 = _local_search_discrete(np.ones(N), max_iter=3000, 
                                       time_limit=min(30, DEADLINE - TIME_MARGIN - (time.time() - start_time)))
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Strategy 2: Try Thue-Morse sequence
    if time.time() < DEADLINE - TIME_MARGIN:
        tm = _thue_morse_sequence(N)
        tm = np.where(tm == 0, 1.0, -1.0)
        c3_tm = _max_sidelobe(_compute_autocorr(tm))
        if c3_tm < best_c3:
            best_x = tm.copy()
            best_c3 = c3_tm
        
        # Polish Thue-Morse with local search
        if time.time() < DEADLINE - TIME_MARGIN:
            remaining = DEADLINE - TIME_MARGIN - time.time()
            x_tm, c3_tm2 = _local_search_discrete(tm, max_iter=2000, time_limit=min(20, remaining))
            if c3_tm2 < best_c3:
                best_x = x_tm
                best_c3 = c3_tm2
    
    # Strategy 3: Simulated Annealing from multiple initializations
    inits = [
        np.ones(N),
        np.random.choice([-1.0, 1.0], size=N),
        _thue_morse_sequence(N),
        np.random.choice([-1.0, 1.0], size=N),
    ]
    
    for trial, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining < 10:
            break
        
        # Simulated Annealing
        x_sa, c3_sa = _simulated_annealing(x_init, time_limit=min(25, remaining))
        if c3_sa < best_c3:
            best_x = x_sa
            best_c3 = c3_sa
        
        # Polish with discrete local search
        if time.time() < DEADLINE - TIME_MARGIN:
            remaining2 = DEADLINE - TIME_MARGIN - time.time()
            x3, c3_3 = _local_search_discrete(best_x, max_iter=2000, time_limit=min(15, remaining2))
            if c3_3 < best_c3:
                best_x = x3
                best_c3 = c3_3
    
    # Strategy 4: Gradient-based search from the best current solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 10:
            x4, c3_4 = _gradient_based_search(best_x, beta=20.0, maxiter=25, maxfun=80,
                                               time_limit=min(20, remaining))
            if c3_4 < best_c3:
                best_x = x4
                best_c3 = c3_4
            
            # Polish with local search
            if time.time() < DEADLINE - TIME_MARGIN:
                remaining2 = DEADLINE - TIME_MARGIN - time.time()
                x5, c3_5 = _local_search_discrete(best_x, max_iter=2000, time_limit=min(15, remaining2))
                if c3_5 < best_c3:
                    best_x = x5
                    best_c3 = c3_5
    
    # Strategy 5: Higher beta gradient search for sharper optimization
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 10:
            x6, c3_6 = _gradient_based_search(best_x, beta=50.0, maxiter=20, maxfun=60,
                                               time_limit=min(15, remaining))
            if c3_6 < best_c3:
                best_x = x6
                best_c3 = c3_6
            
            # Final polish
            if time.time() < DEADLINE - TIME_MARGIN:
                remaining2 = DEADLINE - TIME_MARGIN - time.time()
                x7, c3_7 = _local_search_discrete(best_x, max_iter=2000, time_limit=remaining2)
                if c3_7 < best_c3:
                    best_x = x7
                    best_c3 = c3_7
    
    # Final polish with longer local search
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 5:
            x8, c3_8 = _local_search_discrete(best_x, max_iter=3000, time_limit=remaining)
            if c3_8 < best_c3:
                best_x = x8
                best_c3 = c3_8
    
    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    x = np.sign(x)
    x[x == 0] = 1
    return x.astype(int)
# EVOLVE-BLOCK-END
