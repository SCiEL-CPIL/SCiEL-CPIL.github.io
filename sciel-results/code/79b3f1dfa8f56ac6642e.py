import numpy as np

def construct():
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    M = len(primes)
    
    if M == 0:
        return []
        
    G = limit # Max possible gap
    
    # 2. Initialize arrays
    # dp[i][g] is not stored globally, only locally for each i
    # pref_val[i][g] = max length ending at i with last gap <= g
    # pref_arg[i][g] = the specific gap k <= g that achieved pref_val[i][g]
    # prev_idx[i][g] = index j such that dp[i][g] came from dp[j][pref_arg[j][g]]
    # prev_gap[i][g] = the specific gap k at j that was used
    
    pref_val = np.zeros((M, G + 1), dtype=np.int16)
    pref_arg = np.zeros((M, G + 1), dtype=np.int16)
    prev_idx = np.full((M, G + 1), -1, dtype=np.int16)
    prev_gap = np.full((M, G + 1), -1, dtype=np.int16)
    
    # 3. DP
    for i in range(M):
        pi = primes[i]
        # Local dp for current i
        dp_i = np.zeros(G + 1, dtype=np.int16)
        
        # Base case: sequence of length 1, last gap 0
        dp_i[0] = 1
        
        # Transition from all j < i
        for j in range(i):
            pj = primes[j]
            g = pi - pj
            # g is the new gap
            # We need max length ending at j with last gap <= g
            best_len_j = pref_val[j, g]
            if best_len_j > 0:
                new_len = best_len_j + 1
                if new_len > dp_i[g]:
                    dp_i[g] = new_len
                    prev_idx[i, g] = j
                    # The gap at j that gave best_len_j is pref_arg[j, g]
                    prev_gap[i, g] = pref_arg[j, g]
        
        # Compute prefix max for i
        # pref_val[i, 0] = dp_i[0]
        # pref_arg[i, 0] = 0
        pref_val[i, 0] = dp_i[0]
        pref_arg[i, 0] = 0
        
        for g in range(1, G + 1):
            if dp_i[g] > pref_val[i, g - 1]:
                pref_val[i, g] = dp_i[g]
                pref_arg[i, g] = g
            else:
                pref_val[i, g] = pref_val[i, g - 1]
                pref_arg[i, g] = pref_arg[i, g - 1]
                
    # 4. Find best ending
    # The max length ending at i is pref_val[i, G]
    max_len = 0
    best_i = -1
    for i in range(M):
        if pref_val[i, G] > max_len:
            max_len = pref_val[i, G]
            best_i = i
            
    if best_i == -1:
        return []
        
    # 5. Backtrack
    # The specific gap at best_i is pref_arg[best_i, G]
    curr_i = best_i
    curr_g = pref_arg[best_i, G]
    
    path = []
    while curr_i != -1:
        path.append(primes[curr_i])
        if curr_g == 0:
            break
        next_i = prev_idx[curr_i, curr_g]
        next_g = prev_gap[curr_i, curr_g]
        curr_i = next_i
        curr_g = next_g
        
    path.reverse()
    return path.tolist()
