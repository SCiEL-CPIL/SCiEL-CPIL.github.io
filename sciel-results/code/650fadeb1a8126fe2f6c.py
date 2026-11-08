import numpy as np
import time

def construct():
    """
    Strategy: Simulated Annealing on a discrete grid of step heights.
    
    The problem asks for step heights h on [0, 2] with integral 1.
    Let N be the number of steps. Width w = 2/N.
    Integral = sum(h_i) * w = 1 => sum(h_i) = N/2.
    
    We model h_i as taking values in [0, 1].
    To minimize the maximum overlap I(t) = sum(h_i * (1 - h_{(i-t) mod N}) * w),
    we effectively want to minimize the autocorrelation A(t) = sum(h_i * h_{(i-t) mod N}).
    Since I(t) = 1 - A(t) (assuming integral 1 and periodicity), minimizing max I(t)
    is equivalent to maximizing min A(t).
    
    The "Erdos" optimal solution for such problems often involves a "random-like" or "low-discrepancy" distribution of high and low values, rather than a smooth wave.
    Specifically, if we can make A(t) as flat as possible, the max overlap is minimized.
    
    We use Simulated Annealing to perturb the sequence of heights.
    To handle the exact sum constraint sum(h) = N/2, we will maintain a binary-like state
    with a fixed number of "1"s and "0"s (or close to it), and then project to [0,1]
    with the exact sum.
    
    Actually, a simpler and more robust approach for the "step height" constraint:
    We can treat the heights as continuous variables but use a discrete perturbation strategy.
    However, the optimal solution for minimizing max overlap in such additive combinatorics contexts
    is often achieved by a "balanced" sequence.
    
    Let's try a specific construction first:
    If we set h_i to be 0 or 1, sum(h) = N/2 means exactly N/2 ones and N/2 zeros.
    This is a binary sequence.
    The overlap I(t) for binary h is:
    I(t) = sum( h_i * (1 - h_{i-t}) ) * w
    If h is binary, h_i(1-h_{i-t}) is 1 if h_i=1 and h_{i-t}=0, else 0.
    So I(t) counts the number of "1->0" transitions at shift t, scaled by w.
    
    We want to minimize the maximum number of such transitions over all shifts t.
    This is related to the "circular autocorrelation" of a binary sequence.
    
    Let's use Simulated Annealing to find a binary sequence of length N with N/2 ones
    that minimizes the maximum circular cross-correlation with its complement.
    
    N must be even. Let's use N=100 as in the seed.
    N/2 = 50 ones, 50 zeros.
    
    Objective: Maximize the minimum autocorrelation A(t) = sum(h_i * h_{i-t}).
    Note: For binary sequences with fixed weight, the autocorrelation is related to the overlap.
    Actually, let's just compute the overlap directly to be safe.
    
    Overlap(t) = sum_{i=0}^{N-1} h_i * (1 - h_{(i-t) % N}) * (2/N)
    
    We want to minimize max_t Overlap(t).
    
    Algorithm:
    1. Initialize a random binary sequence with 50 ones and 50 zeros.
    2. Compute max overlap.
    3. Simulated Annealing:
       - Pick a random index i.
       - Pick a random index j where h_j != h_i.
       - Swap h_i and h_j.
       - Compute new max overlap (or approximate change).
       - Accept if better, or with probability exp(-delta/T) if worse.
    4. Return the sequence.
    
    To speed up, we can compute the overlap spectrum using FFT.
    Overlap(t) is proportional to the circular cross-correlation of h and (1-h).
    Let H = FFT(h), G = FFT(1-h).
    Cross-correlation C = IFFT(H * conj(G)).
    The real part of C gives the correlation values.
    
    Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 100
    target_sum = N / 2.0  # 50.0
    
    # Initialize with a random binary sequence
    # We need exactly 50 ones and 50 zeros.
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    # Function to compute max overlap using FFT
    def compute_max_overlap(h_seq):
        # h_seq is a numpy array of length N
        # We want I(t) = sum h_i (1 - h_{i-t})
        # This is the cross-correlation of h and (1-h).
        # Note: Standard correlation is sum h_i * g_{i-t}.
        # Here g = 1 - h.
        # So we want corr(h, 1-h).
        
        # Using FFT for circular correlation
        H = np.fft.fft(h_seq)
        G = np.fft.fft(1.0 - h_seq)
        
        # Correlation is IFFT(H * conj(G))
        corr = np.fft.ifft(H * np.conj(G)).real
        
        # The overlap integral is sum(h_i * (1-h_{i-t})) * (2/N)
        # The correlation array 'corr' contains sum(h_i * (1-h_{(i-t) mod N}))
        # So Overlap(t) = corr[t] * (2/N)
        
        return np.max(corr) * (2.0 / N)
    
    current_overlap = compute_max_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Pick two random distinct indices
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        if i == j:
            continue
            
        # Only swap if values are different (otherwise no change)
        if h[i] == h[j]:
            continue
            
        # Perform swap
        h[i], h[j] = h[j], h[i]
        
        new_overlap = compute_max_overlap(h)
        delta = new_overlap - current_overlap
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_overlap = new_overlap
            if new_overlap < best_overlap:
                best_overlap = new_overlap
                best_h = h.copy()
        else:
            # Revert swap
            h[i], h[j] = h[j], h[i]
            
        T *= cooling_rate
        
    # Ensure the sum is exactly N/2
    # Since we started with binary and only swapped, sum is preserved exactly as N/2.
    # However, the task allows values in [0,1].
    # The binary solution is a valid step height sequence.
    # We return the sequence.
    
    # Double check constraints
    if not np.all((best_h >= 0) & (best_h <= 1)):
        # Fallback to constant 0.5 if something went wrong
        best_h = np.full(N, 0.5)
        
    # The sum should be exactly N/2.
    # In floating point, 50.0 is exact.
    
    return best_h

# EVOLVE-BLOCK-END
