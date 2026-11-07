import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Generate blocks (p, a, T, k_block)
    # T = p^a * sigma(p^a)
    # k_block = p^a
    
    blocks = []
    
    # Primes up to 200
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    for p in primes:
        p_pow = p
        sigma_val = 1 + p
        while True:
            T = p_pow * sigma_val
            if T > 10**18:
                break
            blocks.append((p, p_pow, T))
            # Next exponent
            p_pow *= p
            sigma_val = sigma_val * p + 1
            # Check overflow or limit
            if p_pow > 10**18:
                break
            
    # Sort blocks by T to find small M first? Or just process all.
    # Processing order doesn't matter for correctness, but might affect performance if we prune.
    
    # M_to_ks: dict mapping M to list of k
    M_to_ks = defaultdict(list)
    M_to_ks[1].append(1)
    
    max_count = 1
    best_M = 1
    
    # We need to be careful with time.
    # Number of blocks is roughly sum of log_p(10^18).
    # For p=2, ~60. For p=3, ~37. For p=5, ~25.
    # Total blocks ~ a few hundred.
    
    # Iterative expansion
    # We iterate through blocks. For each block, we extend all existing solutions.
    # To avoid using the same prime twice, we check if k is divisible by p.
    
    # Optimization: If time is running out, return best so far.
    
    # We can also limit the number of k's stored per M to 1000.
    
    for p, k_block, T in blocks:
        # Snapshot of current solutions to avoid modifying while iterating
        # But we are adding to new M's, so we can iterate over a list of items
        # However, M_to_ks is a defaultdict, so we should get a list of items first
        # Or better, we can just iterate over a copy of the items?
        # Copying is expensive.
        # Alternative: maintain a list of (M, k) pairs?
        # Let's stick to dict but be careful.
        # Actually, we can just iterate over list(M_to_ks.items())
        
        items = list(M_to_ks.items())
        
        for M, ks in items:
            # Check time
            if time.time() > deadline:
                break
                
            M_new = M * T
            if M_new > 10**18:
                continue
                
            # Check if we already have 1000 solutions for M_new
            if len(M_to_ks[M_new]) >= 1000:
                continue
                
            # Generate new k's
            for k in ks:
                if k % p == 0:
                    continue
                k_new = k * k_block
                M_to_ks[M_new].append(k_new)
                
                # Update max count
                cnt = len(M_to_ks[M_new])
                if cnt > max_count:
                    max_count = cnt
                    best_M = M_new
                    
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break
            
    if max_count == 1:
        return {'M': 1, 'k': [1]}
        
    # Return the best M and its k's
    # Ensure k's are distinct (they should be, as k is built from unique primes)
    # But just in case, convert to set then list
    k_list = list(set(M_to_ks[best_M]))
    
    # Sort k_list for determinism? Not required, but good.
    k_list.sort()
    
    return {'M': int(best_M), 'k': k_list}
