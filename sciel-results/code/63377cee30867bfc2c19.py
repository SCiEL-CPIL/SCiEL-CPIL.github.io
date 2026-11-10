"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    M = 10**6 + 24
    
    # Precompute p_part for all numbers up to M
    # p_part[j] is the product of prime powers p^k where p in P and p^k || j
    p_part = [1] * (M + 1)
    
    # Efficient precomputation
    for j in range(1, M + 1):
        x = j
        part = 1
        for p in P:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                    part *= p
        p_part[j] = part
        
        # Time check during precomputation (should be fast, but just in case)
        if time.time() > deadline - 5.0:
            # Fallback to a simple guess if precomputation is too slow (unlikely)
            return 1000

    best_n = 1000
    best_score = 9 # Max distinct primes is 9, but we count distinct values, so max 24.
    # Actually max distinct values is 24.
    best_score = 24
    
    # Scan n
    for n in range(1000, 10**6 + 1):
        # Check time
        if time.time() > deadline:
            break
            
        seen = set()
        # Unroll inner loop for speed? 24 is small.
        for i in range(1, 25):
            v = p_part[n + i]
            if v not in seen:
                seen.add(v)
                if len(seen) >= best_score:
                    break
        
        current_score = len(seen)
        if current_score < best_score:
            best_score = current_score
            best_n = n
            # Minimum possible score is 2 (as derived above)
            if best_score == 2:
                break

    return best_n

# EVOLVE-BLOCK-END
