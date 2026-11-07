"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _is_valid(els):
    """Check if all non-empty subset sums are distinct and pairwise incomparable under divisibility."""
    n = len(els)
    if n == 0:
        return False
    
    # Generate all subset sums iteratively
    # Start with {0}
    sums_set = {0}
    current_sums = [0]
    
    for e in els:
        new_sums = []
        for s in current_sums:
            ns = s + e
            if ns in sums_set:
                return False
            sums_set.add(ns)
            new_sums.append(ns)
        current_sums.extend(new_sums)
    
    # Remove 0 for divisibility check
    sums_set.discard(0)
    if not sums_set:
        return False
        
    sorted_sums = sorted(sums_set)
    max_sum = sum(els)
    
    for a in sorted_sums:
        # Check if any multiple of a is in sums_set
        # We only need to check up to max_sum
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += a
            
    return True

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Try to find a valid set of increasing size
    # Start from a reasonable maximum and go down
    # Size 10-12 is likely feasible with random large numbers
    for target_size in [12, 11, 10, 9, 8]:
        if time.time() - start_time > 50:
            break
            
        # Number of attempts depends on size
        # Larger sizes are harder, so fewer attempts if we are running out of time
        max_attempts = 2000 if target_size <= 10 else 500
        
        for _ in range(max_attempts):
            if time.time() - start_time > 55:
                return [100, 101, 103] # Fallback
            
            # Sample from a range that reduces divisibility likelihood
            # Large numbers make it harder for small sums to divide large sums
            # unless the large sum is exactly a multiple, which is less likely
            # with random large primes/numbers.
            cand = random.sample(range(100, 257), target_size)
            
            if _is_valid(cand):
                return [int(x) for x in cand]
                
    # Fallback to a known small valid set
    # {100, 101, 103}
    # Sums: 100, 101, 103, 201, 203, 204, 304
    # Check divisibility:
    # 100: 200 (no), 300 (no)
    # 101: 202 (no), 303 (no)
    # 103: 206 (no), 309 (no)
    # 201: 402 (no)
    # 203: 406 (no)
    # 204: 408 (no)
    # 304: 608 (no)
    # Distinct? Yes.
    return [100, 101, 103]
# EVOLVE-BLOCK-END
