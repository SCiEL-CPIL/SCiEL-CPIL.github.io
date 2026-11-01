"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 145.0  # Safety margin against the 150s hard kill
TIME_MARGIN = 5.0


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    # We only need the positive lags 1 to N-1 due to symmetry
    sidelobes = np.abs(R[1:N])
    max_sidelobe = np.max(sidelobes)
    return max_sidelobe / R0


def _local_search_greedy(x, max_iter=2000, time_limit=10.0):
    """
    Greedy local search that specifically targets the lag with the maximum sidelobe.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    
    # Precompute n_fft for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Get the current autocorrelation to find the worst lag
        F = np.fft.rfft(x, n=n_fft)
        power = F * np.conj(F)
        R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        R0 = R[0]
        
        # Find the lag with the maximum absolute sidelobe
        sidelobes = np.abs(R[1:N])
        worst_lag = np.argmax(sidelobes) + 1
        
        # Try flipping bits that are likely to reduce the contribution to R[worst_lag]
        # R[k] = sum_{i} x[i] x[i+k]
        # Flipping x[j] changes R[k] by -2 * x[j] * (sum of neighbors)
        # Heuristic: try flipping random bits, but prioritize those near the edges of the correlation window?
        # Simpler: just try a few random flips and check if the worst lag improves.
        
        improved = False
        for _ in range(10): # Try 10 random flips
            if time.time() > DEADLINE - TIME_MARGIN:
                return best_x, best_c3
            
            idx = random.randint(0, N - 1)
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            
            # Quick check: compute C3
            F_new = np.fft.rfft(x_new, n=n_fft)
            power_new = F_new * np.conj(F_new)
            R_new = np.fft.irfft(power_new, n=n_fft)[:2 * N - 1]
            new_c3 = np.max(np.abs(R_new[1:N])) / R0
            
            if new_c3 < current_c3 - 1e-12:
                x = x_new
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_x = x.copy()
                    best_c3 = current_c3
                improved = True
                break
        
        if not improved:
            # If no improvement in 10 tries, stop or continue? 
            # For greedy, we might stop early to save time for SA.
            pass
            
    return best_x, best_c3


def _simulated_annealing(x, max_iter=50000, T_start=0.5, T_end=0.01, time_limit=None):
    """
    Simulated annealing with adaptive multi-flip moves.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Move: flip 1 or 2 bits
        n_flips = 1 if random.random() < 0.7 else 2
        idxs = random.sample(range(N), n_flips)
        
        x_new = x.copy()
        for idx in idxs:
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


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    tm = np.zeros(n)
    for i in range(n):
        c = 0
        m = i
        while m:
            c += m & 1
            m >>= 1
        tm[i] = 1.0 if c % 2 == 0 else -1.0
    return tm


def _random_phase_sequence(n, seed=None):
    """
    Generate a sequence with random phases, then map to +/- 1.
    This often has better initial autocorrelation properties than pure random +/- 1.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    phases = rng.uniform(0, 2*np.pi, n)
    # Map phase to +/- 1 based on cosine sign
    return np.sign(np.cos(phases))


def _construct():
    """
    Main construction function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # 1. Try some structured initializations
    inits = [
        _thue_morse(N),
        _random_phase_sequence(N, seed=1),
        _random_phase_sequence(N, seed=2),
        np.random.choice([-1, 1], size=N),
        np.random.choice([-1, 1], size=N)
    ]
    
    # 2. Run Simulated Annealing from each initialization
    # Allocate time budget for SA
    sa_time_per_init = 15.0
    
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        x_trial, c3_trial = _simulated_annealing(
            x_init.copy(), 
            max_iter=20000, 
            T_start=0.5, 
            T_end=0.005, 
            time_limit=sa_time_per_init
        )
        
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # 3. Final Polish: Greedy Local Search on the best result
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        x_polished, c3_polished = _local_search_greedy(
            best_x.copy(), 
            max_iter=5000, 
            time_limit=remaining_time
        )
        if c3_polished < best_c3:
            best_x = x_polished
            best_c3 = c3_polished

    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
