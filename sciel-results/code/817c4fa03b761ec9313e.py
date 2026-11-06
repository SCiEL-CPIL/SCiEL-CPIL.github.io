import math
import time

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Precomputed optimal solution:
    # A = [4, 16, 40], B = [1, 3, 7]
    # Sums:
    # 4+1=5, 4+3=7, 4+7=11
    # 16+1=17, 16+3=19, 16+7=23
    # 40+1=41, 40+3=43, 40+7=47
    # All 9 sums are distinct primes, hence pairwise coprime.
    # Cardinality is 9 (maximum possible for 3x3 sets with distinct sums).
    
    A = [4, 16, 40]
    B = [1, 3, 7]
    
    # Verification (optional, but good for safety)
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    
    # Check if all distinct sums are pairwise coprime
    # Since they are primes, this should pass.
    # If for some reason they aren't primes (they are), this check ensures validity.
    valid = True
    if len(distinct_sums) != 9:
        valid = False
    else:
        for i in range(9):
            for j in range(i + 1, 9):
                if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return [A, B]
        
    # Fallback if the hardcoded solution somehow fails (it shouldn't)
    # Use a known valid smaller solution or seed
    return [[1, 7, 13], [4, 10, 16]]
