"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Optimal selections for each prime based on subset sum distinctness of exponents
    # p=2: powers 2,4,8,16,32 (exp 1,2,3,4,5). Best size 3: {1,2,4} -> 2,4,16
    # p=3: powers 3,9,27 (exp 1,2,3). Best size 2: {1,2} -> 3,9
    # p=5: powers 5,25 (exp 1,2). Best size 2: {1,2} -> 5,25
    # p>=7: only p. Size 1.
    
    result = []
    
    # Add optimized sets for small primes
    result.extend([2, 4, 16])
    result.extend([3, 9])
    result.extend([5, 25])
    
    # Add remaining primes
    for p in primes:
        if p > 5:
            result.append(p)
            
    # Verify validity (safety check)
    # This check is fast for 16 elements (2^16 = 65536 products)
    if len(result) > 0:
        prods = set()
        # Use iterative doubling or simple recursion
        # Simple recursion for clarity, 16 elements is small
        def gen_prods(idx, current):
            if idx == len(result):
                if current in prods:
                    return False
                prods.add(current)
                return True
            # Don't pick
            if not gen_prods(idx + 1, current):
                return False
            # Pick
            if not gen_prods(idx + 1, current * result[idx]):
                return False
            return True
        
        if not gen_prods(0, 1):
            # Fallback to primes if collision found (should not happen)
            return primes
            
    return result
# EVOLVE-BLOCK-END
