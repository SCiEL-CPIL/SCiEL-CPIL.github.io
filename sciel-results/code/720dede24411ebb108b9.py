import numpy as np
import time

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Generate primes <= 10000
    def sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    max_gap = int(primes[-1] - primes[0])
    
    # Use int16 to save memory. Max length is small, max index is < 1229, max gap is < 10000.
    # Note: int16 max is 32767.
    
    # dp[i][g]: length of longest valid subseq ending at i with last gap g
    dp = np.zeros((n, max_gap + 1), dtype=np.int16)
    # best_up_to[i][g]: max(dp[i][0..g])
    best_up_to = np.zeros((n, max_gap + 1), dtype=np.int16)
    # argmax_prefix[i][g]: the gap g' in [0..g] that achieves best_up_to[i][g]
    argmax_prefix = np.zeros((n, max_gap + 1), dtype=np.int16)
    # parent_idx[i][g]: index j of the previous prime
    parent_idx = np.full((n, max_gap + 1), -1, dtype=np.int16)
    # parent_gap[i][g]: the gap g' used at j
    parent_gap = np.zeros((n, max_gap + 1), dtype=np.int16)
    
    # Initialize for i=0
    dp[0, 0] = 1
    best_up_to[0, 0] = 1
    argmax_prefix[0, 0] = 0
    for g in range(1, max_gap + 1):
        # Since dp[0, g] is 0 for g>0, best_up_to[0, g] remains 1
        # argmax_prefix[0, g] should point to 0
        best_up_to[0, g] = 1
        argmax_prefix[0, g] = 0

    max_len = 1
    best_end_i = 0
    best_end_g = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        # Base case: sequence of length 1 starting at i
        dp[i, 0] = 1
        
        pi = int(primes[i])
        
        for j in range(i):
            g = pi - int(primes[j])
            if g > max_gap:
                continue
            
            # Get best length ending at j with last gap <= g
            prev_len = int(best_up_to[j, g])
            candidate = prev_len + 1
            
            if candidate > dp[i, g]:
                dp[i, g] = candidate
                parent_idx[i, g] = j
                parent_gap[i, g] = int(argmax_prefix[j, g])
                
                if candidate > max_len:
                    max_len = candidate
                    best_end_i = i
                    best_end_g = g
        
        # Compute prefix max and argmax for row i
        # We need to update best_up_to[i] and argmax_prefix[i]
        # best_up_to[i, g] = max(dp[i, 0..g])
        
        # We can do this iteratively or use numpy
        # Iterative is O(G) per i, which is fine.
        
        current_max = 0
        current_argmax = 0
        for g in range(max_gap + 1):
            val = int(dp[i, g])
            if val > current_max:
                current_max = val
                current_argmax = g
            best_up_to[i, g] = current_max
            argmax_prefix[i, g] = current_argmax
            
    # Reconstruct
    seq = []
    i = best_end_i
    g = best_end_g
    
    # Safety check: if no valid sequence found (shouldn't happen), return first prime
    if max_len == 1:
         # Find any i with dp[i][0] == 1
         # Actually best_end_i, best_end_g might be 0,0
         seq.append(int(primes[best_end_i]))
         return seq
         
    while i >= 0 and len(seq) < n + 1:
        seq.append(int(primes[i]))
        if parent_idx[i, g] == -1:
            break
        i = int(parent_idx[i, g])
        g = int(parent_gap[i, g])
        
    seq.reverse()
    
    return seq
