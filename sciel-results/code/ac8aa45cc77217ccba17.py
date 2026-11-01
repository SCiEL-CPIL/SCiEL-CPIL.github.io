"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 120.0
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k != 0} |R[k]| / R[0]
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


def _local_search(x, max_iter=20000, time_limit=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
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


def _simulated_annealing(x, max_iter=50000, T_start=1.0, T_end=0.0001, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    
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


def _multi_flip_search(x, max_iter=10000, time_limit=None, n_flips=1):
    """
    Search that allows flipping multiple bits at once.
    """
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
        
        indices = random.sample(range(N), min(n_flips, N))
        x_new = x.copy()
        for idx in indices:
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


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Local search from all ones
    x1, c3_1 = _local_search(np.ones(N), max_iter=5000, time_limit=15)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 2: Random initializations + local search
    for trial in range(10):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=3000, time_limit=8)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    remaining = DEADLINE - TIME_MARGIN - time.time()
    if remaining > 10:
        x3, c3_3 = _simulated_annealing(best_x, max_iter=30000, T_start=0.5, T_end=0.0001,
                                         time_limit=min(remaining * 0.4, 40))
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 4: Multi-flip search
    remaining = DEADLINE - TIME_MARGIN - time.time()
    if remaining > 5:
        x4, c3_4 = _multi_flip_search(best_x, max_iter=5000, time_limit=min(remaining * 0.3, 20), n_flips=2)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    # Final local search polish
    remaining = DEADLINE - TIME_MARGIN - time.time()
    if remaining > 2:
        x5, c3_5 = _local_search(best_x, max_iter=10000, time_limit=remaining)
        if c3_5 < best_c3:
            best_x = x5
            best_c3 = c3_5
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
