"""Spectral optimization for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 150  # seconds, with safety margin


def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h) over all shifts.
    Overlap at shift k is sum_i h[i] * (1 - h[(i-k) % N]).
    We want to minimize the maximum of these overlaps.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    one_minus_h = 1.0 - h
    
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    
    corr = np.real(np.fft.ifft(np.conj(H) * O))
    
    return np.max(corr)


def construct():
    """
    Construct step heights h on [0,1] with sum = N/2, minimizing max overlap with shifted 1-h.
    
    Strategy: 
    1. Start from binary sequences (0/1) since they have higher variance and potentially
       better autocorrelation properties.
    2. Use simulated annealing to explore the space of binary sequences with exactly N/2 ones.
    3. The objective is to minimize the maximum overlap, which is equivalent to 
       maximizing the minimum autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 10  # 10 second safety margin
    
    def _score(h):
        """Compute max overlap for a sequence."""
        return _compute_max_overlap(h)
    
    def _random_binary_seq():
        """Generate a random binary sequence with exactly N/2 ones."""
        seq = np.zeros(N, dtype=np.float64)
        ones = np.random.choice(N, N//2, replace=False)
        seq[ones] = 1.0
        return seq
    
    # Best solution found so far
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try a few initial patterns
    np.random.seed(42)
    
    # Pattern 1: Alternating 1,0,1,0,...
    seq_alt = np.zeros(N, dtype=np.float64)
    seq_alt[::2] = 1.0
    score_alt = _score(seq_alt)
    if score_alt < best_score:
        best_score = score_alt
        best_h = seq_alt.copy()
    
    # Pattern 2: First half ones, second half zeros
    seq_half = np.zeros(N, dtype=np.float64)
    seq_half[:N//2] = 1.0
    score_half = _score(seq_half)
    if score_half < best_score:
        best_score = score_half
        best_h = seq_half.copy()
    
    # Simulated annealing
    T = 2.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9999
    
    current_h = _random_binary_seq()
    current_score = _score(current_h)
    
    if current_score < best_score:
        best_score = current_score
        best_h = current_h.copy()
    
    while time.time() < deadline:
        # Generate a new candidate by swapping a 1 and a 0
        ones = np.where(current_h == 1.0)[0]
        zeros = np.where(current_h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        # Swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        
        new_score = _score(current_h)
        
        # Accept if better or with some probability
        if new_score < current_score:
            # Always accept improvements
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if delta > 0:
                accept_prob = np.exp(-delta / T)
                if np.random.rand() < accept_prob:
                    current_score = new_score
                else:
                    # Revert the swap
                    current_h[i], current_h[j] = current_h[j], current_h[i]
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final local search: try all single swaps to see if we can improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(best_h == 1.0)[0]
        zeros = np.where(best_h == 0.0)[0]
        
        for i in ones:
            for j in zeros:
                if time.time() > deadline:
                    break
                
                # Try swapping
                best_h[i], best_h[j] = best_h[j], best_h[i]
                new_score = _score(best_h)
                
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    best_h[i], best_h[j] = best_h[j], best_h[i]
            
            if time.time() > deadline:
                break
    
    return best_h
