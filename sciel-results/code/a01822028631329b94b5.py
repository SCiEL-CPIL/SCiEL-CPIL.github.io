"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_sidelobe(h):
    """
    Compute the peak sidelobe of the linear autocorrelation.
    r(tau) = sum_{i=0}^{N-1-tau} h[i] * h[i+tau] for tau = 0, 1, ..., N-1
    We want max |r(tau)| for tau != 0.
    """
    n = len(h)
    # Use FFT for linear autocorrelation
    size = 2 * n
    H = np.fft.fft(h, n=size)
    R = np.fft.ifft(H * np.conj(H)).real
    # R[0] is r(0) = sum h[i]^2 = N (for ±1)
    # R[1:N] are r(1), r(2), ..., r(N-1)
    # R[N:2N-1] are the negative lags (symmetric for real signals)
    # We only need positive lags 1..N-1
    sidelobes = R[1:n]
    return np.max(np.abs(sidelobes))


def construct():
    """
    Construct a signed step function (400 intervals) to minimize the peak sidelobe
    of the linear autocorrelation (upper bound on C3).
    
    Uses simulated annealing with multiple restarts and different initializations.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def get_score(h):
        return _compute_peak_sidelobe(h)

    best_h = None
    best_score = float('inf')

    def try_improve(h, max_time=30.0):
        """Run simulated annealing starting from h, return best found."""
        nonlocal best_h, best_score, start_time, deadline
        
        current_h = h.copy()
        current_score = get_score(current_h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
        
        # Simulated annealing parameters
        T = max(1.0, current_score * 0.1)
        cooling_rate = 0.9999
        min_T = 1e-8
        local_T = time.time()
        
        # Precompute indices for fast flip evaluation
        idx = np.arange(N)
        
        while time.time() < deadline - 5.0 and T > min_T:
            # Try flipping each element once in random order
            order = np.random.permutation(N)
            
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                
                # Flip element i
                old_val = current_h[i]
                current_h[i] = -old_val
                
                new_score = get_score(current_h)
                delta = new_score - current_score
                
                # Accept if better, or with probability exp(-delta/T)
                if delta < 0 or (T > 1e-10 and np.exp(-delta / T) > np.random.random()):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                else:
                    # Revert
                    current_h[i] = old_val
            
            T *= cooling_rate
        
        return current_h, current_score

    # Multiple initialization strategies
    rng = np.random.RandomState(42)
    
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1.0
    initializations.append(alt)
    
    # 3. Random
    initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # 4. Block-based with various block sizes
    for block_size in [10, 20, 50, 100]:
        h_block = np.zeros(N)
        for start in range(0, N, block_size):
            sign = rng.choice([-1.0, 1.0])
            end = min(start + block_size, N)
            h_block[start:end] = sign
        initializations.append(h_block)
    
    # 5. More random initializations
    for _ in range(3):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # Try each initialization with simulated annealing
    for init_h in initializations:
        if time.time() > deadline - 10.0:
            break
        try_improve(init_h)
    
    # Final polish: greedy local search on best found
    if best_h is not None and time.time() < deadline - 3.0:
        current_h = best_h.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                old_val = current_h[i]
                current_h[i] = -old_val
                new_score = get_score(current_h)
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_h = current_h.copy()
                    improved = True
                else:
                    current_h[i] = old_val
    
    return best_h
# EVOLVE-BLOCK-END
