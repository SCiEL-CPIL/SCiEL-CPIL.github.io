import math
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    def is_dissociated(S):
        sums = [0]
        for x in S:
            val = L // x
            new_sums = [s + val for s in sums]
            sums.extend(new_sums)
        return len(set(sums)) == len(sums)
    
    # Try to find a large dissociated set
    # Check contiguous sets from the end
    best = []
    for start in range(1, 23):
        S = list(range(start, 23))
        if len(S) > len(best):
            if is_dissociated(S):
                best = S
            else:
                # If a large set fails, smaller sets starting earlier might fail too?
                # Not necessarily. But we can break if len(S) <= len(best)
                if len(S) <= len(best):
                    break
    
    if best:
        return best
        
    # Fallback: Randomized search or greedy
    # For now, return a known good set if best is empty
    # But best should not be empty.
    return [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
