import time
import numpy as np

def _is_prime_sieve(n):
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 85
    
    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
        
    max_gap = N_LIMIT
    
    # dp[j][g] = max length ending at prime j with last gap g
    # PMax[j][g] = max_{h<=g} dp[j][h]
    # argmax_PMax[j][g] = argmax_{h<=g} dp[j][h]
    # parent_i[j][g] = index i of previous prime
    # parent_gap[j][g] = gap h of previous prime's state
    
    # Using int16 to save memory and potentially speed up cache usage
    # Max length is small, max index is < 2000, max gap is 10000.
    # int16 holds up to 32767. Sufficient.
    
    dp = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    PMax = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    argmax_PMax = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    parent_i = [np.full(max_gap + 1, -1, dtype=np.int16) for _ in range(n_primes)]
    parent_gap = [np.full(max_gap + 1, -1, dtype=np.int16) for _ in range(n_primes)]
    
    p_arr = np.array(primes, dtype=np.int32)
    
    # Initialize base cases: "sequence of length 1" ending at each prime with gap 0
    for j in range(n_primes):
        dp[j][0] = 1
        PMax[j][0] = 1
        argmax_PMax[j][0] = 0
        # parent_i[j][0] remains -1
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
            
        if j > 0:
            p_j = p_arr[j]
            dp_j = dp[j]
            pi_j = parent_i[j]
            pg_j = parent_gap[j]
            
            # Iterate over previous primes
            # Accessing numpy arrays in Python loop
            for i in range(j):
                g = int(p_j - p_arr[i])
                
                # Get best predecessor
                prev_max = PMax[i][g]
                
                if prev_max > 0:
                    cand = prev_max + 1
                    arg_h = argmax_PMax[i][g]
                else:
                    # Should not happen if base case is set, but safe
                    cand = 2
                    arg_h = 0
                
                if cand > dp_j[g]:
                    dp_j[g] = cand
                    pi_j[g] = i
                    pg_j[g] = arg_h
            
            # Compute PMax[j] and argmax_PMax[j]
            PMax[j] = np.maximum.accumulate(dp[j])
            
            # Compute argmax_PMax[j] in a loop
            # This is O(MaxGap) per j
            running_max = 0
            running_arg = 0
            argmax_j = argmax_PMax[j]
            dp_j_ref = dp[j]
            for g in range(1, max_gap + 1):
                v = dp_j_ref[g]
                if v > running_max:
                    running_max = v
                    running_arg = g
                argmax_j[g] = running_arg

    # Find global max
    max_len = 0
    best_j = -1
    best_gap = -1

    for j in range(n_primes):
        row_max = np.max(dp[j])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(dp[j]))

    if max_len < 2:
        return [primes[0]]

    # Reconstruct
    result = []
    cur_j = best_j
    cur_gap = best_gap

    while cur_j >= 0:
        result.append(primes[cur_j])
        pi = parent_i[cur_j][cur_gap]
        pg = parent_gap[cur_j][cur_gap]

        if pi < 0:
            break

        cur_j = int(pi)
        cur_gap = int(pg)

    result.reverse()
    
    # Verification
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Fallback
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values
            
    return result
