"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect

def _sieve_primes(limit):
    """Return numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    primes_list = primes.tolist()
    
    # DP approach:
    # dp[t][s] = length of longest valid subsequence ending at primes[t] with immediate predecessor primes[s]
    #           where s < t
    # The gap is primes[t] - primes[s]
    # Constraint: the gap before primes[s] (if it exists) must be <= primes[t] - primes[s]
    #
    # Transition:
    #   dp[t][s] = 2 if s == 0 (no predecessor for primes[0])
    #            = 1 + max{dp[s][r] : r < s, primes[s]-primes[r] <= primes[t]-primes[s]} if s > 0
    #            = 2 if no valid r exists
    #
    # For each s, we precompute a data structure to answer:
    #   Query(g): max{dp[s][r] : r < s, primes[s]-primes[r] <= g}
    # and which r achieved it.
    #
    # We store for each s: sorted list of (gap_sr, dp[s][r], r) and prefix max with argmax.
    
    # s_data[s] = (sorted_gaps, prefix_max_dp, prefix_argmax_r) or None if s==0
    s_data = [None] * n
    
    # dp_table[t] = list of dp[t][s] for s in range(t)
    dp_table = [None] * n
    # parent[t][s] = r that achieved the max, or -1
    parent = [None] * n
    
    for t in range(n):
        if time.time() > deadline:
            break
        if t == 0:
            dp_table[t] = []
            parent[t] = []
            s_data[t] = None
            continue
        
        dp_t = [0] * t
        parent_t = [-1] * t
        
        for s in range(t):
            gap_ts = primes_list[t] - primes_list[s]
            
            if s == 0:
                dp_t[s] = 2
                parent_t[s] = -1
            else:
                data_s = s_data[s]
                if data_s is None:
                    dp_t[s] = 2
                    parent_t[s] = -1
                else:
                    sorted_gaps_s, prefix_max_dp_s, prefix_argmax_s = data_s
                    # Binary search for rightmost index where sorted_gaps_s[idx] <= gap_ts
                    idx = bisect.bisect_right(sorted_gaps_s, gap_ts) - 1
                    if idx >= 0:
                        best_dp = prefix_max_dp_s[idx]
                        if best_dp > 0:
                            dp_t[s] = 1 + best_dp
                            parent_t[s] = prefix_argmax_s[idx]
                        else:
                            dp_t[s] = 2
                            parent_t[s] = -1
                    else:
                        dp_t[s] = 2
                        parent_t[s] = -1
        
        dp_table[t] = dp_t
        parent[t] = parent_t
        
        # Build s_data[t] for future queries
        entries = [(primes_list[t] - primes_list[s], dp_t[s], s) for s in range(t)]
        entries.sort(key=lambda x: x[0])
        sorted_gaps = [e[0] for e in entries]
        dp_vals = [e[1] for e in entries]
        r_vals = [e[2] for e in entries]
        
        prefix_max = [0] * t
        prefix_argmax = [-1] * t
        cur_max = 0
        cur_arg = -1
        for i in range(t):
            if dp_vals[i] > cur_max:
                cur_max = dp_vals[i]
                cur_arg = r_vals[i]
            prefix_max[i] = cur_max
            prefix_argmax[i] = cur_arg
        
        s_data[t] = (sorted_gaps, prefix_max, prefix_argmax)
    
    # Find the best (t, s) pair
    best_val = 1
    best_t = 0
    best_s = -1
    for t in range(n):
        if dp_table[t] is None:
            continue
        for s in range(t):
            if dp_table[t][s] > best_val:
                best_val = dp_table[t][s]
                best_t = t
                best_s = s
    
    # Reconstruct the sequence
    if best_s == -1:
        return [primes_list[best_t]]
    
    # Chain: best_t, best_s, parent[best_t][best_s], parent[best_s][parent[best_t][best_s]], ...
    seq_indices = [best_t, best_s]
    ct = best_t
    cs = best_s
    while parent[ct] is not None and parent[ct][cs] != -1:
        ns = parent[ct][cs]
        seq_indices.append(ns)
        ct = cs
        cs = ns
    
    seq_indices.reverse()
    result = [primes_list[i] for i in seq_indices]
    
    # Verify correctness
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        valid = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
        if not valid:
            # Fallback: return a simple valid sequence
            return [primes_list[0]]
    
    return result

# EVOLVE-BLOCK-END
