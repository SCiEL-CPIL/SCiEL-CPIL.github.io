import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Generate primes <= 10000
    is_prime = [True] * 10001
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(10000) + 1):
        if is_prime[i]:
            for j in range(i*i, 10001, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    # dp[j] is a list of states for prime j
    # Each state: (gap, length, prev_idx, prev_gap)
    # prev_idx is the index of the previous prime in the subsequence
    # prev_gap is the gap value in the previous state that allowed this transition
    # If it's the start of the sequence, prev_idx = -1, prev_gap = 0 (or -1)
    
    dp = [[] for _ in range(n)]
    
    for j in range(n):
        # Start a new sequence at j
        # Gap is effectively 0 for the first step? 
        # The condition is g_new >= g_old.
        # If sequence is just [p_j], there is no gap yet.
        # If we append p_k, gap is p_k - p_j. Condition: p_k - p_j >= 0. Always true.
        # So initial state for j is (0, 1, -1, 0)
        states = [(0, 1, -1, 0)]
        
        for i in range(j):
            # Check time
            if time.time() > deadline:
                # Fallback to greedy or best found so far
                # We need to return something valid.
                # Let's return the best sequence found so far in dp
                break
                
            for state in dp[i]:
                g_old, l_old, prev_idx, prev_gap = state
                g_new = primes[j] - primes[i]
                if g_new >= g_old:
                    # We can extend
                    # New state: (g_new, l_old + 1, i, g_old)
                    states.append((g_new, l_old + 1, i, g_old))
        
        if time.time() > deadline:
             # Fallback
             pass
             
        # Prune states for j
        # Sort by gap, then by length descending
        states.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_len_so_far = 0
        # We iterate through states sorted by gap.
        # A state (g, l) is dominated if there exists a state (g', l') with g' <= g and l' >= l.
        # Since sorted by g, g' <= g is satisfied by all previous states.
        # So we just need to check if l > max_len_so_far.
        # Wait, if l == max_len_so_far, it is dominated (same length, larger gap is worse or equal).
        # Actually, if g' < g and l' == l, (g', l') dominates (g, l).
        # So we keep state only if l > max_len_so_far.
        
        for s in states:
            g, l, pi, pg = s
            if l > max_len_so_far:
                pruned.append(s)
                max_len_so_far = l
                
        dp[j] = pruned
        
        if time.time() > deadline:
            break

    # Find the best state
    best_len = 0
    best_j = -1
    best_state = None
    for j in range(n):
        for state in dp[j]:
            if state[1] > best_len:
                best_len = state[1]
                best_j = j
                best_state = state
                
    if best_j == -1:
        return [primes[0]] if primes else []
        
    # Reconstruct
    seq = []
    cur_j = best_j
    cur_state = best_state
    while cur_j != -1:
        seq.append(primes[cur_j])
        g, l, pi, pg = cur_state
        if pi == -1:
            break
        # Find the state in dp[pi] that matches (pg, l-1)
        # There might be multiple states with same gap and length?
        # We pruned, so gaps are unique for a given "max_len" progression?
        # Actually, in pruned list, for a specific gap, we only kept the one that wasn't dominated.
        # But we stored the specific `pg` (gap in previous step) and `pi` (index).
        # We need to find the state in dp[pi] that has gap == pg and length == l-1.
        target_gap = pg
        target_len = l - 1
        prev_state = None
        for s in dp[pi]:
            if s[0] == target_gap and s[1] == target_len:
                prev_state = s
                break
        if prev_state is None:
            # This shouldn't happen if logic is correct
            break
        cur_state = prev_state
        cur_j = pi
        
    seq.reverse()
    return seq

# Fallback if deadline exceeded during DP
# The loop above breaks, but dp might be incomplete.
# We should ensure we return a valid sequence.
# The code above handles this by finding the best in the partially filled dp.
# However, if we break early, dp[j] for j > current might be empty.
# The best found so far is still valid.

# Let's add a greedy fallback if DP is too slow or fails?
# The DP is O(N^2 K). N=1229.
# If K is small, it's fast.
# If it times out, we return the best found so far.
# To be safe, we can implement a greedy solution as a baseline and return max(greedy, dp).
# But the task says "Maximize length". Greedy is not optimal.
# We will stick to DP with deadline.
