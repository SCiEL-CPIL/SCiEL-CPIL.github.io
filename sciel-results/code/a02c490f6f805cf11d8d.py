"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_sidelobe_max(x):
    """Compute max absolute autocorrelation value for non-zero lags."""
    # Linear autocorrelation
    n = len(x)
    # Use FFT for efficiency
    size = 2 * n
    X = np.fft.fft(x, size=size)
    R = np.fft.ifft(X * np.conj(X)).real
    # R[0] is the peak (n * sum(x^2) = n since x is ±1)
    # Sidelobes are R[1:n] and R[-n+1:-1] which are R[n+1:size]
    # For linear autocorrelation, lags 1 to n-1 are R[1:n]
    # and lags -(n-1) to -1 are R[n+1:size] (but these are conjugate symmetric for real)
    sidelobes = np.concatenate([R[1:n], R[n+1:size]])
    return np.max(np.abs(sidelobes))

def _compute_score(x):
    """The score is the upper bound on C3 - we want to minimize this."""
    return _compute_sidelobe_max(x)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_score = _compute_score(x)
    
    # Local search: try flipping individual bits
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        idx = np.random.randint(N)
        x[idx] *= -1
        new_score = _compute_score(x)
        
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Revert
            x[idx] *= -1
    
    # Try a few different random restarts if time permits
    while time.time() < deadline - 5:
        x = np.random.choice([-1, 1], size=N)
        cur_score = _compute_score(x)
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()
        
        # Local search from this start
        local_iter = 0
        while time.time() < deadline - 5 and local_iter < 2000:
            idx = np.random.randint(N)
            x[idx] *= -1
            new_score = _compute_score(x)
            if new_score < cur_score:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                x[idx] *= -1
            local_iter += 1
    
    return best_x
# EVOLVE-BLOCK-END
