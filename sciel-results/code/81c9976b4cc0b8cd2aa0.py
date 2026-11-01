"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
DEADLINE = 145.0
TIME_MARGIN = 5.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k != 0} |R[k]| / R[0]
    where R is the autocorrelation.
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


def _compute_c3_fast(x, R0=None):
    """
    Fast computation of C3, optionally reusing R0.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    if R0 is None:
        R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _compute_c3_soft(x, beta=10.0):
    """
    Compute a smooth approximation of C3 using log-sum-exp for the max operation.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    max_val = np.max(abs_sidelobes)
    if max_val == 0:
        return 0.0
    
    log_sum_exp = np.log(np.sum(np.exp(beta * (abs_sidelobes - max_val)))) + beta * max_val
    soft_max = log_sum_exp / beta
    
    C3_soft = soft_max / R0
    return C3_soft


def _local_search(x, max_iter=5000, time_limit=None, start_time=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
    if start_time is None:
        start_time = time.time()
    current_c3 = _compute_c3(x)
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
        new_c3 = _compute_c3(x_new)
        
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


def _multi_flip_search(x, max_iter=3000, flips_per_step=2, time_limit=None, start_time=None):
    """
    Multi-flip local search: try flipping 2-3 positions at once.
    """
    if start_time is None:
        start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 300
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        n_flips = random.randint(1, flips_per_step)
        indices = random.sample(range(N), n_flips)
        x_new = x.copy()
        for i in indices:
            x_new[i] = -x_new[i]
        new_c3 = _compute_c3(x_new)
        
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


def _simulated_annealing(x, max_iter=20000, T_start=1.0, T_end=0.01, time_limit=None, start_time=None):
    """
    Simulated annealing to escape local minima.
    """
    if start_time is None:
        start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _gradient_descent_soft(x, max_iter=500, lr=0.1, beta=10.0, time_limit=None, start_time=None):
    """
    Gradient-based optimization using smooth approximation of C3.
    Uses finite differences to estimate gradients.
    """
    if start_time is None:
        start_time = time.time()
    current_x = x.astype(float)
    current_c3 = _compute_c3_soft(current_x, beta)
    best_x = x.copy()
    best_c3 = _compute_c3(x)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Compute gradient using finite differences
        grad = np.zeros(N)
        for i in range(N):
            x_plus = current_x.copy()
            x_plus[i] += 1.0
            x_minus = current_x.copy()
            x_minus[i] -= 1.0
            
            f_plus = _compute_c3_soft(x_plus, beta)
            f_minus = _compute_c3_soft(x_minus, beta)
            
            grad[i] = (f_plus - f_minus) / 2.0
        
        # Update in direction of negative gradient
        current_x = current_x - lr * grad
        
        # Project back to ±1
        current_x = np.sign(current_x)
        current_x[current_x == 0] = 1
        
        current_c3 = _compute_c3_soft(current_x, beta)
        exact_c3 = _compute_c3(current_x.astype(int))
        
        if exact_c3 < best_c3:
            best_x = current_x.astype(int)
            best_c3 = exact_c3
        
        # Reduce learning rate
        lr *= 0.99
    
    return best_x, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Start with all ones, do local search
    x1, c3_1 = _local_search(np.ones(N), max_iter=3000, time_limit=25, start_time=start_time)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 2: Random initialization + local search
    for trial in range(5):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=2000, time_limit=12, start_time=start_time)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Multi-flip search from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _multi_flip_search(best_x, max_iter=2000, flips_per_step=3, 
                                       time_limit=remaining_time * 0.3, start_time=start_time)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 4: Gradient-based optimization with soft surrogate
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _gradient_descent_soft(best_x, max_iter=300, lr=0.5, beta=20.0, 
                                           time_limit=remaining_time * 0.3, start_time=start_time)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 5: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        x5, c3_5 = _simulated_annealing(best_x, max_iter=8000, T_start=0.5, T_end=0.01,
                                         time_limit=remaining_time * 0.5, start_time=start_time)
        if c3_5 < best_c3:
            best_x = x5
            best_c3 = c3_5
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        x6, c3_6 = _local_search(best_x, max_iter=3000, time_limit=remaining_time, start_time=start_time)
        if c3_6 < best_c3:
            best_x = x6
            best_c3 = c3_6
    
    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
