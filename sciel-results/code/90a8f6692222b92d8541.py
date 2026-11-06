import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build dense incidence matrix M of shape (n, Np)
    # M[i, pid] = 1 if position i is in progression pid, else 0
    M = np.zeros((n, Np), dtype=np.int8)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1
    
    def compute_sums(signs):
        # signs is (n,) vector of -1, 1
        # sums is (Np,) vector of signed sums
        # sums = signs @ M
        return signs @ M
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n)
    sums = compute_sums(signs)
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    current_score = best_score
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # Multi-restart SA
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            pass # Already initialized
        else:
            signs = rng.choice([-1, 1], size=n)
            sums = compute_sums(signs)
            current_score = max_abs_sum(sums)
            T = T_start
        
        if current_score < best_score:
            best_signs = signs.copy()
            best_score = current_score
        
        T = T_start
        
        # SA Loop
        # We want to run until time runs out or T < T_end
        # Each step: pick random position, flip, compute delta
        # To speed up, we can batch some operations or just rely on vectorization
        # However, flipping one sign requires updating sums.
        # sums_new = sums + 2 * new_sign * M[i, :]
        # This is a vector operation of size Np.
        
        while time.time() < deadline - 5 and T > T_end:
            i = rng.randint(n)
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Update sums
            # Delta for each progression p is 2 * new_sign * M[i, p]
            # sums_new = sums + 2 * new_sign * M[i, :]
            # Since M[i, :] is int8, convert to int32 for matmul or just use array ops
            # M[i] is shape (Np,)
            delta_vec = 2 * new_sign * M[i].astype(np.int32)
            sums_new = sums + delta_vec
            
            new_score = max_abs_sum(sums_new)
            
            delta = new_score - current_score
            
            if delta < 0:
                # Accept
                signs[i] = new_sign
                sums = sums_new
                current_score = new_score
                if current_score < best_score:
                    best_signs = signs.copy()
                    best_score = current_score
            else:
                # Probabilistic accept
                # P = exp(-delta / T)
                # To avoid underflow, if delta/T > 50, P ~ 0
                if delta < 50 * T:
                    p = np.exp(-delta / T)
                    if rng.random_sample() < p:
                        signs[i] = new_sign
                        sums = sums_new
                        current_score = new_score
            
            T *= cooling_rate
            
            # Optional: occasionally check best
            # Not strictly necessary as we update best on improvement
        
    # Final polish: try all single flips to see if any improves best_score
    # Use the best_signs found
    if time.time() < deadline - 2:
        signs = best_signs.copy()
        sums = compute_sums(signs)
        current_score = max_abs_sum(sums)
        
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            delta_vec = 2 * new_sign * M[i].astype(np.int32)
            sums_new = sums + delta_vec
            new_score = max_abs_sum(sums_new)
            
            if new_score < current_score:
                signs[i] = new_sign
                sums = sums_new
                current_score = new_score
                best_signs = signs.copy()
                best_score = current_score
            else:
                signs[i] = old_sign
                # sums remains unchanged
    
    return [int(x) for x in best_signs]
