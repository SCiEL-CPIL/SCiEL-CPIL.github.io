import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    dp = [None] * n
    dp[0] = [(1, 0, -1, -1)]
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    MAX_L_BUDGET = 100 # Safe upper bound for length
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        # Initialize buckets with base case: sequence of length 1 ending at i
        # L=1, G=0, prev_i=-1, prev_state_idx=-1
        min_gap_for_L = [float('inf')] * (MAX_L_BUDGET + 1)
        state_for_L = [None] * (MAX_L_BUDGET + 1)
        
        min_gap_for_L[1] = 0
        state_for_L[1] = (1, 0, -1, -1)
        
        actual_max_L = 1
        
        for j in range(i):
            gap = primes[i] - primes[j]
            states = dp[j]
            if not states:
                continue
            
            # Check if smallest gap in states is > gap
            # states is sorted by G descending (index 0 is largest G, last is smallest)
            if states[-1][1] > gap:
                continue
            
            # Scan from end (smallest G) to start (largest G)
            for k in range(len(states) - 1, -1, -1):
                L, G, pi, si = states[k]
                if G <= gap:
                    new_L = L + 1
                    if new_L > MAX_L_BUDGET:
                        continue # Ignore too long sequences (should not happen)
                    
                    if new_L > actual_max_L:
                        actual_max_L = new_L
                        
                    if G < min_gap_for_L[new_L]:
                        min_gap_for_L[new_L] = G
                        state_for_L[new_L] = (new_L, G, j, k)
                else:
                    break
        
        # Prune using buckets
        non_dom = []
        current_min_gap = float('inf')
        
        for L in range(actual_max_L, 0, -1):
            if state_for_L[L] is not None:
                cand = state_for_L[L]
                G = cand[1]
                if G < current_min_gap:
                    non_dom.append(cand)
                    current_min_gap = G
                    
        dp[i] = non_dom
        
        if non_dom:
            current_best_len = non_dom[0][0]
            if current_best_len > best_len:
                best_len = current_best_len
                best_i = i
                best_state_idx = 0

    # Reconstruct
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq
