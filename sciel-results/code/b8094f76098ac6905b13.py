"""Combinatorial greedy construction + local search for ±1 sequences minimizing max|autoconv|/S^2."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _score(x):
    """Compute max|autoconv| / S^2 for a ±1 sequence."""
    s = np.sum(x)
    if s == 0:
        return 1e10
    ac = _autoconv_fft(x)
    m = np.max(np.abs(ac))
    return m / (s * s)

def _greedy_construct(rng, sign_bias=0.0):
    """
    Greedy construction: build the sequence one element at a time,
    choosing ±1 to minimize the current peak autoconvolution contribution.
    """
    x = np.zeros(N)
    
    for i in range(N):
        # Try both signs
        best_val = 1e10
        best_sign = 1
        
        for sign in [1, -1]:
            x[i] = sign + sign_bias
            s = np.sum(x)
            if s == 0:
                x[i] = 0
                continue
            # Compute autoconvolution for the partial sequence
            # Only need to check lags that are affected
            ac = _autoconv_fft(x)
            m = np.max(np.abs(ac))
            val = m / (s * s)
            if val < best_val:
                best_val = val
                best_sign = sign
        
        x[i] = best_sign
    
    return x

def _local_search(x, n_iterations=2000, rng=None):
    """
    Local search: flip individual bits or small sets of bits
    to reduce the peak autoconvolution.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    x = x.copy()
    current_score = _score(x)
    
    for it in range(n_iterations):
        if _check_deadline():
            break
        
        # Choose a random set of positions to flip
        n_flip = rng.integers(1, 4)  # 1 to 3 positions
        idx = rng.choice(N, size=n_flip, replace=False)
        
        x_trial = x.copy()
        x_trial[idx] *= -1
        
        trial_score = _score(x_trial)
        
        if trial_score < current_score:
            current_score = trial_score
            x = x_trial
    
    return x, current_score

def _simulated_annealing(x, n_iterations=5000, rng=None):
    """
    Simulated annealing: accept worse solutions with decreasing probability
    to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    
    x = x.copy()
    current_score = _score(x)
    best_score = current_score
    best_x = x.copy()
    
    T_start = 0.01
    T_end = 0.0001
    cooling = (T_end / T_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        if _check_deadline():
            break
        
        T = T_start * (cooling ** it)
        
        # Flip 1-3 random bits
        n_flip = rng.integers(1, 4)
        idx = rng.choice(N, size=n_flip, replace=False)
        
        x_trial = x.copy()
        x_trial[idx] *= -1
        
        trial_score = _score(x_trial)
        delta = trial_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = trial_score
            x = x_trial
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    return best_x, best_score

def _multi_start_search(n_starts=10, rng=None):
    """
    Run multiple independent searches from different initializations
    and return the best result.
    """
    if rng is None:
        rng = np.random.default_rng(999)
    
    best_x = None
    best_score = 1e10
    
    for start in range(n_starts):
        if _check_deadline():
            break
        
        # Different initialization strategies
        if start == 0:
            # Greedy construction
            x0 = _greedy_construct(rng)
        elif start == 1:
            # Random balanced sequence
            x0 = rng.choice([-1, 1], size=N)
            # Ensure roughly balanced
            while np.sum(x0) > N * 0.1:
                flip_idx = np.where(x0 == 1)[0]
                if len(flip_idx) == 0:
                    break
                x0[flip_idx[rng.integers(len(flip_idx))]] = -1
        elif start == 2:
            # Alternating with noise
            x0 = np.ones(N)
            x0[::2] = -1
            noise_idx = rng.choice(N, size=N//10, replace=False)
            x0[noise_idx] *= -1
        elif start == 3:
            # Block pattern
            block_size = N // 8
            x0 = np.tile(np.array([1, 1, -1, -1]), N // 4)
            x0 = x0[:N]
        elif start == 4:
            # Sine-based
            t = np.linspace(0, 2*np.pi, N)
            x0 = np.sign(np.sin(t) + 0.3 * np.sin(2*t))
            x0[x0 == 0] = 1
        elif start == 5:
            # Random walk
            steps = rng.choice([-1, 1], size=N)
            x0 = np.sign(np.cumsum(steps))
            x0[x0 == 0] = 1
        elif start == 6:
            # Thue-Morse-like
            x0 = np.ones(N)
            for i in range(1, N):
                x0[i] = -x0[i >> 1]
        elif start == 7:
            # Mersenne-like pattern
            x0 = np.ones(N)
            for i in range(1, N):
                if (i & -i) == i:  # power of 2
                    x0[i] = -1
        elif start == 8:
            # Random with bias toward balance
            x0 = rng.choice([-1, 1], size=N, p=[0.49, 0.51])
        else:
            # Pure random
            x0 = rng.choice([-1, 1], size=N)
        
        # Run local search
        x_ls, score_ls = _local_search(x0, n_iterations=500, rng=rng)
        
        # Run simulated annealing from the local search result
        x_sa, score_sa = _simulated_annealing(x_ls, n_iterations=2000, rng=rng)
        
        if score_sa < best_score:
            best_score = score_sa
            best_x = x_sa.copy()
    
    return best_x, best_score

def construct():
    """
    Main entry point: construct a ±1 sequence minimizing max|autoconv|/S^2.
    Returns the sequence as a numpy array.
    """
    rng = np.random.default_rng(2024)
    
    # Multi-start combinatorial search
    best_x, best_score = _multi_start_search(n_starts=8, rng=rng)
    
    # Final polish with more iterations
    if not _check_deadline():
        best_x, best_score = _local_search(best_x, n_iterations=1000, rng=rng)
    
    return best_x
# EVOLVE-BLOCK-END
