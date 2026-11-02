import numpy as np
import time

N = 400

def _score(s):
    """
    Compute the upper bound on C3: max absolute off-peak autocorrelation normalized by N.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.fft(s_f, n=n_fft)
    acf = np.fft.ifft(F * np.conj(F)).real[:N]
    acf /= N
    return float(np.max(np.abs(acf[1:])))

def construct():
    """
    Construct a signed step function (sequence of +/-1) of length 400 
    that minimizes the max absolute off-peak autocorrelation.
    """
    deadline = time.time() + 110  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initialize with a random sequence
    best_seq = rng.choice([-1, 1], size=N).astype(np.int8)
    best_score = _score(best_seq)
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.9995
    
    current_seq = best_seq.copy()
    current_score = best_score
    
    steps_since_improve = 0
    max_steps_no_improve = 5000
    
    # We will try to flip blocks of size 1, 2, or 3
    block_sizes = [1, 1, 1, 2, 2, 3] # Weighted towards single flips for fine tuning
    
    while time.time() < deadline:
        # Select a block size
        b_size = rng.choice(block_sizes)
        
        # Select random start index
        start_idx = rng.integers(N - b_size + 1)
        
        # Apply flip
        original_block = current_seq[start_idx:start_idx+b_size].copy()
        current_seq[start_idx:start_idx+b_size] *= -1
        
        new_score = _score(current_seq)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq.copy()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            # Revert
            current_seq[start_idx:start_idx+b_size] = original_block
            steps_since_improve += 1
            
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Restart logic
        if steps_since_improve > max_steps_no_improve:
            # Perturb the best solution found so far
            current_seq = best_seq.copy()
            for _ in range(5):
                idx = rng.integers(N)
                current_seq[idx] *= -1
            current_score = _score(current_seq)
            steps_since_improve = 0
            T = 0.005 # Reset temperature

    return best_seq
