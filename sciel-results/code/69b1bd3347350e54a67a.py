import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate initial Thue-Morse sequence
    # Thue-Morse: t(0)=0, t(n) = t(floor(n/2)) XOR (n mod 2)
    def thue_morse(n):
        t = 0
        while n > 0:
            t ^= n & 1
            n >>= 1
        return t
        
    h = np.array([thue_morse(i) for i in range(N)], dtype=float)
    
    # Ensure sum is N/2
    # Thue-Morse for N=100 might not have exactly 50 ones.
    # Count ones
    num_ones = np.sum(h)
    target_ones = N / 2
    
    # Adjust to have exactly N/2 ones
    # If num_ones > target, flip some 1s to 0s
    # If num_ones < target, flip some 0s to 1s
    # Prefer flipping bits that are "central" or random to maintain structure?
    # Let's just flip random bits until sum is correct.
    np.random.seed(42)
    indices_ones = np.where(h == 1)[0]
    indices_zeros = np.where(h == 0)[0]
    
    if num_ones > target_ones:
        to_flip = int(num_ones - target_ones)
        flip_idx = np.random.choice(indices_ones, to_flip, replace=False)
        h[flip_idx] = 0
    elif num_ones < target_ones:
        to_flip = int(target_ones - num_ones)
        flip_idx = np.random.choice(indices_zeros, to_flip, replace=False)
        h[flip_idx] = 1
        
    def score(h):
        # Calculate max overlap for linear shifts
        # O_k = sum_{i=0}^{N-1-k} h_i (1-h_{i+k})
        max_ov = 0
        for k in range(N):
            if k == 0:
                # sum h_i (1-h_i)
                ov = np.sum(h * (1 - h))
            else:
                ov = np.sum(h[:N-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    current_score = score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = 1.0 - old_val
        
        new_score = score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
        else:
            # Revert
            h[idx] = old_val
            
        T *= cooling
        if T < 1e-6:
            T = 1.0 # Restart or break? Let's just keep going with low T
            
    return best_h
# EVOLVE-BLOCK-END
