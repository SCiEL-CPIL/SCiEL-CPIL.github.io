import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    limit = 10000
    # Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
        
    # Map prime value to index, -1 if not prime
    p_to_idx = np.full(limit + 1, -1, dtype=np.int32)
    p_to_idx[primes] = np.arange(N)
    
    # DP tables
    best = np.zeros((N, limit + 1), dtype=np.uint8)
    argmax_gap = np.zeros((N, limit + 1), dtype=np.uint16)
    parent_idx = np.full((N, limit + 1), -1, dtype=np.int32)
    
    global_max_len = 1
    best_end_i = 0
    best_end_g = 0
    
    # Base case: i=0
    best[0, :] = 1
    argmax_gap[0, :] = 0
    
    for i in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
            
        pi = primes[i]
        max_g = pi - 2
        
        exact = np.zeros(limit + 1, dtype=np.uint8)
        exact[0] = 1
        
        if max_g >= 1:
            g_vals = np.arange(1, max_g + 1)
            prev_vals = pi - g_vals
            mask = is_prime[prev_vals]
            if np.any(mask):
                valid_g = g_vals[mask]
                valid_prev = prev_vals[mask]
                valid_j = p_to_idx[valid_prev]
                
                prev_lengths = best[valid_j, valid_g]
                new_lengths = prev_lengths + 1
                exact[valid_g] = new_lengths
                
                parent_exact = np.full(limit + 1, -1, dtype=np.int32)
                parent_exact[valid_g] = valid_j
                
                best_i = np.maximum.accumulate(exact)
                argmax_i = np.zeros(limit + 1, dtype=np.uint16)
                parent_i = np.full(limit + 1, -1, dtype=np.int32)
                
                current_max = 0
                current_arg = 0
                current_parent = -1
                for g in range(limit + 1):
                    if exact[g] > current_max:
                        current_max = exact[g]
                        current_arg = g
                        current_parent = parent_exact[g]
                    argmax_i[g] = current_arg
                    parent_i[g] = current_parent
                
                best[i, :] = best_i
                argmax_gap[i, :] = argmax_i
                parent_idx[i, :] = parent_i
                
                if best_i[-1] > global_max_len:
                    global_max_len = best_i[-1]
                    best_end_i = i
                    best_end_g = argmax_i[-1]
        
    # Reconstruct
    if global_max_len == 1:
        return [int(primes[0])]
        
    seq = []
    cur_i = best_end_i
    cur_g = best_end_g
    
    while cur_i >= 0:
        seq.append(int(primes[cur_i]))
        if cur_i == 0:
            break
        next_i = parent_idx[cur_i, cur_g]
        if next_i == -1:
            break
        # The next state's gap is the argmax for the parent at the current gap?
        # No.
        # State (i, g) came from (j, g_prev).
        # We stored parent_idx[i, g] = j.
        # But which gap g_prev did j have?
        # It was the gap that achieved best[j, g].
        # We stored argmax_gap[j, g] as the specific gap k such that exact[j, k] = best[j, g].
        # So the next state is (j, argmax_gap[j, g]).
        
        next_g = argmax_gap[cur_i, cur_g] # Wait.
        
        # Let's re-verify the reconstruction logic.
        # We are at state (i, g_i).
        # Value is L.
        # If L == 1, stop.
        # Else, it came from j = parent_idx[i, g_i].
        # The value at j was best[j, g_i] (because exact[i, g_i] = best[j, g_i] + 1).
        # Wait, exact[i, g] is defined for the *current* gap g.
        # The parent state is (j, g_prev) where g_prev is the gap that achieved the max at j.
        # Specifically, best[j, g] = max_{k<=g} exact[j, k].
        # Let k* be the argmax.
        # Then the path goes through (j, k*).
        # So the next state in the sequence is (j, k*).
        # k* is stored in argmax_gap[j, g].
        
        # So:
        # next_i = parent_idx[cur_i, cur_g]
        # next_g = argmax_gap[next_i, cur_g]
        
        # Let's trace:
        # cur_i, cur_g
        # next_i = parent_idx[cur_i, cur_g]
        # next_g = argmax_gap[next_i, cur_g]
        
        # Is this correct?
        # exact[cur_i, cur_g] = best[next_i, cur_g] + 1.
        # best[next_i, cur_g] is the max length ending at next_i with last gap <= cur_g.
        # This max is achieved at gap next_g = argmax_gap[next_i, cur_g].
        # So the previous prime is next_i, and its "effective" last gap in the optimal chain is next_g.
        # So yes, next state is (next_i, next_g).
        
        cur_i = next_i
        cur_g = next_g
        
    seq.reverse()
    return seq
