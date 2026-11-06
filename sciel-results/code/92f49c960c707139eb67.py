import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill

    LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
            
    primes = np.where(is_prime)[0].tolist()
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    MAX_D = LIMIT
    
    # Initialize DP arrays
    # pref[i][d] = max length of valid sequence ending at prime i with last gap <= d
    # argmax[i][d] = the gap value k <= d that achieves pref[i][d]
    pref = np.zeros((N, MAX_D + 1), dtype=np.int32)
    argmax = np.zeros((N, MAX_D + 1), dtype=np.int32)
    
    # Backpointers
    # back_idx[i][d] = index j of previous prime in optimal sequence for state (i, d)
    # back_gap[i][d] = gap d' used at prime j
    back_idx = np.full((N, MAX_D + 1), -1, dtype=np.int32)
    back_gap = np.full((N, MAX_D + 1), -1, dtype=np.int32)
    
    global_max_len = 1
    best_end_i = 0
    best_end_d = 0 # Corresponds to the gap that led to best_end_i, or 0 if start
    
    for i in range(N):
        # Temp array for dp values for prime i
        # dp_curr[d] = length of valid sequence ending at prime i with last gap EXACTLY d
        dp_curr = np.zeros(MAX_D + 1, dtype=np.int32)
        dp_curr[0] = 1  # Base case: sequence [primes[i]]
        
        p_i = primes[i]
        
        # Iterate over all previous primes j
        for j in range(i):
            d = p_i - primes[j]
            # d is the new gap
            # We need max length ending at j with gap <= d
            max_prev_len = pref[j][d]
            best_prev_gap = argmax[j][d]
            
            new_len = max_prev_len + 1
            dp_curr[d] = new_len
            
            # Store backpointers
            back_idx[i][d] = j
            back_gap[i][d] = best_prev_gap
            
            # Check global max
            if new_len > global_max_len:
                global_max_len = new_len
                best_end_i = i
                best_end_d = d
                
        # Build prefix max and argmax for row i
        current_max = 0
        current_argmax = 0
        for d in range(MAX_D + 1):
            if dp_curr[d] > current_max:
                current_max = dp_curr[d]
                current_argmax = d
            pref[i][d] = current_max
            argmax[i][d] = current_argmax
            
        # Update global max if the best sequence ends at i with a smaller gap than MAX_D?
        # pref[i][MAX_D] is the max length ending at i.
        # We already updated global_max_len inside the j-loop for specific d.
        # But what if the max length ending at i is from a d that wasn't a transition?
        # dp_curr[0] = 1. If i=0, global_max_len becomes 1.
        # If i>0, and no transitions improved global max, but pref[i][MAX_D] > global_max_len?
        # This can't happen because any sequence ending at i with gap d>0 was considered in the j-loop.
        # The only sequence not considered in the j-loop is the single element [p_i] (gap 0).
        # Length 1. Global max is at least 1.
        
        # Time check
        if time.time() > DEADLINE:
            break

    # Reconstruct sequence
    if best_end_i == -1:
        return [primes[0]]
        
    seq = []
    i = best_end_i
    d = best_end_d
    
    while i != -1:
        seq.append(primes[i])
        prev_i = back_idx[i][d]
        prev_d = back_gap[i][d]
        
        if prev_i == -1:
            break
            
        i = prev_i
        d = prev_d
        
    seq.reverse()
    return seq
