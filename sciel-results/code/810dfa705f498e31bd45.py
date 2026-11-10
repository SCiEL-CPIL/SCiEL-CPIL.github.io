# EVOLVE-BLOCK-START
import sys

def construct():
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # up to 1000024
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part[i] = product of p^e for all p in small_primes dividing i
    small_part = [1] * (MAX_VAL + 1)
    for p in small_primes:
        for multiple in range(p, MAX_VAL + 1, p):
            val = multiple
            while val % p == 0:
                small_part[multiple] *= p
                val //= p
                
    # Map unique small parts to IDs for faster frequency counting
    # We only care about indices 1001 to 1000024
    unique_parts = set(small_part[1001:MAX_VAL + 1])
    part_to_id = {part: i for i, part in enumerate(unique_parts)}
    num_parts = len(unique_parts)
    
    part_id = [0] * (MAX_VAL + 1)
    for i in range(1001, MAX_VAL + 1):
        part_id[i] = part_to_id[small_part[i]]
        
    # Sliding window
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = [0] * num_parts
    distinct = 0
    for i in range(1001, 1001 + WINDOW):
        pid = part_id[i]
        if freq[pid] == 0:
            distinct += 1
        freq[pid] += 1
        
    best_count = distinct
    best_n = 1000
    
    # If we found a very low count, we might still want to check all, 
    # but 5 is likely the minimum possible. 
    # Let's just run through all.
    
    for n in range(1001, MAX_N + 1):
        # Remove n (which is (n-1)+1)
        old_pid = part_id[n]
        freq[old_pid] -= 1
        if freq[old_pid] == 0:
            distinct -= 1
            
        # Add n+WINDOW
        new_pid = part_id[n + WINDOW]
        if freq[new_pid] == 0:
            distinct += 1
        freq[new_pid] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            # Optional: if best_count == 5, we could break if we are confident 4 is impossible.
            # But let's be safe and finish, or break if 5 is found and we assume 4 is impossible?
            # The task says MINIMIZE. If 5 is found, 4 is very unlikely. 
            # However, to be rigorous, we continue. The loop is fast with lists.
            
    return best_n
# EVOLVE-BLOCK-END
