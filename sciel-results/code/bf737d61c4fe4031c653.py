"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try Greedy Algorithm first
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
            
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
            
        d = min_d
        while d in used:
            d += 1
            
        if d > 10000000:
            break
            
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
        
    if num != 0 or len(result) == 0 or len(result) > 256:
        # Fallback to DP
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
        raise RuntimeError("no baseline representation")
    
    # Local search: try to reduce count by merging pairs
    # 1/a + 1/b = 1/c iff c = a*b/(a+b) is an integer
    def try_merge(result, used):
        improved = True
        while improved:
            improved = False
            n = len(result)
            for i in range(n):
                if time.time() > deadline:
                    return result
                for j in range(i+1, n):
                    a, b = result[i], result[j]
                    # Check if 1/a + 1/b = 1/c for some integer c >= 5
                    # 1/a + 1/b = (a+b)/(a*b) = 1/c => c = a*b/(a+b)
                    num = a * b
                    den = a + b
                    if den > 0 and num % den == 0:
                        c = num // den
                        if c >= 5 and c <= 10000000 and c not in used:
                            # Replace i and j with c
                            new_result = []
                            new_used = set()
                            for k in range(n):
                                if k != i and k != j:
                                    new_result.append(result[k])
                                    new_used.add(result[k])
                            new_result.append(c)
                            new_used.add(c)
                            # Verify sum is still 1
                            s = sum(Fraction(1, x) for x in new_result)
                            if s == 1 and len(new_result) < len(result):
                                result = new_result
                                used = new_used
                                improved = True
                                break
                if improved:
                    break
        return result
    
    result = try_merge(result, used)
    
    # Verify final result
    s = sum(Fraction(1, x) for x in result)
    if s == 1 and 1 <= len(result) <= 256:
        return result
    
    # If merge failed verification, return original greedy result
    s_orig = sum(Fraction(1, x) for x in result)
    if s_orig == 1 and 1 <= len(result) <= 256:
        return result
    
    raise RuntimeError("no valid representation")
# EVOLVE-BLOCK-END
