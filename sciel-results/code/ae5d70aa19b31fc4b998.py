import numpy as np
import time
import random

N = 400
N_FFT = 2048 # Next power of 2 >= 3*N - 2 (1198)

def compute_bound(s):
    """
    Compute the max absolute value of the linear triple autocorrelation.
    s: array of length N with values +/- 1.
    """
    # Pad to N_FFT
    # s is already length N. We need to treat it as zero outside.
    # np.fft.fft pads with zeros automatically if n > len(s).
    S = np.fft.fft(s, n=N_FFT)
    C3 = np.fft.ifft(S ** 3).real
    return np.max(np.abs(C3))

def construct():
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for safety
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Type 1: Balanced Random
    for _ in range(10):
        s = np.ones(N)
        flip_idx = np.random.choice(N, N // 2, replace=False)
        s[flip_idx] = -1.0
        candidates.append(s)
        
    # Type 2: Logistic Map (Balanced)
    def gen_logistic(seed):
        x = seed
        s = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            s[i] = 1.0 if x > 0.5 else -1.0
        # Force balance
        # Count positives
        pos_count = np.sum(s == 1.0)
        if pos_count > N // 2:
            # Flip excess positives to negative
            pos_indices = np.where(s == 1.0)[0]
            excess = pos_count - N // 2
            to_flip = np.random.choice(pos_indices, excess, replace=False)
            s[to_flip] = -1.0
        elif pos_count < N // 2:
            neg_indices = np.where(s == -1.0)[0]
            deficit = N // 2 - pos_count
            to_flip = np.random.choice(neg_indices, deficit, replace=False)
            s[to_flip] = 1.0
        return s

    seeds = [np.sqrt(2) % 1, np.pi % 1, np.e % 1, 0.123456, 0.987654]
    for seed in seeds:
        candidates.append(gen_logistic(seed))

    # 2. Quick Optimization of Candidates
    best_bound = float('inf')
    best_signs = None
    
    for s in candidates:
        # Short Greedy/SA
        s_opt = optimize_short(s)
        b = compute_bound(s_opt)
        if b < best_bound:
            best_bound = b
            best_signs = s_opt

    # 3. Main Optimization Loop (Simulated Annealing)
    if best_signs is not None:
        s = best_signs.copy()
        current_bound = best_bound
        T = 1.0
        T_end = 0.001
        alpha = 0.995
        
        # Precompute indices for double flips
        # We will pick random i, j
        
        while T > T_end and time.time() - start_time < time_limit:
            # Choose move type
            if random.random() < 0.7:
                # Single Flip
                i = np.random.randint(N)
                s[i] *= -1.0
                new_bound = compute_bound(s)
                delta = new_bound - current_bound
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_bound = new_bound
                else:
                    s[i] *= -1.0
            else:
                # Double Flip (Swap)
                # Pick random +1 and random -1
                pos_idx = np.where(s == 1.0)[0]
                neg_idx = np.where(s == -1.0)[0]
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    i = pos_idx[np.random.randint(len(pos_idx))]
                    j = neg_idx[np.random.randint(len(neg_idx))]
                    s[i] = -1.0
                    s[j] = 1.0
                    new_bound = compute_bound(s)
                    delta = new_bound - current_bound
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_bound = new_bound
                    else:
                        s[i] = 1.0
                        s[j] = -1.0
            
            T *= alpha
            if current_bound < best_bound:
                best_bound = current_bound
                best_signs = s.copy()

    # 4. Final Polish
    if best_signs is not None and time.time() - start_time < time_limit:
        s = best_signs.copy()
        current_bound = best_bound
        
        # Greedy Single Flip
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                s[i] *= -1.0
                new_bound = compute_bound(s)
                if new_bound < current_bound:
                    current_bound = new_bound
                    improved = True
                else:
                    s[i] *= -1.0
        
        # Greedy Double Flip
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Sample a subset of pairs to avoid O(N^2)
            # Or just do a few random passes
            for _ in range(10):
                pos_idx = np.where(s == 1.0)[0]
                neg_idx = np.where(s == -1.0)[0]
                if len(pos_idx) == 0 or len(neg_idx) == 0: break
                # Random pairs
                for _ in range(20):
                    i = pos_idx[np.random.randint(len(pos_idx))]
                    j = neg_idx[np.random.randint(len(neg_idx))]
                    s[i] = -1.0
                    s[j] = 1.0
                    new_bound = compute_bound(s)
                    if new_bound < current_bound:
                        current_bound = new_bound
                        improved = True
                    else:
                        s[i] = 1.0
                        s[j] = -1.0
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_signs = s

    return best_signs

def optimize_short(s):
    """
    A quick local search to improve a candidate.
    """
    current_bound = compute_bound(s)
    improved = True
    # Limit iterations for this phase
    max_iters = 2000
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Random order to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            s[i] *= -1.0
            new_bound = compute_bound(s)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                s[i] *= -1.0
    return s
