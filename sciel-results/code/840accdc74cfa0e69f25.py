"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    def greedy_egyptian(target_frac, min_den=5, max_den=10000000):
        """Sylvester-style greedy: pick smallest valid denominator >= min_den."""
        num, den = target_frac.numerator, target_frac.denominator
        used = set()
        result = []
        
        while num > 0:
            if time.time() > deadline:
                return None
            
            min_d = math.ceil(den / num)
            if min_d < min_den:
                min_d = min_den
            
            d = min_d
            while d in used:
                d += 1
            
            if d > max_den:
                return None
            
            result.append(d)
            used.add(d)
            
            # remaining = num/den - 1/d = (num*d - den)/(den*d)
            num = num * d - den
            den = den * d
            
            g = math.gcd(num, den)
            if g > 1:
                num //= g
                den //= g
        
        return result if result else None
    
    def try_merge_pair(result, i, j):
        """Try replacing result[i] and result[j] with a single denominator."""
        if i == j:
            return None
        a, b = result[i], result[j]
        # 1/a + 1/b = (a+b)/(a*b)
        # We need 1/c = (a+b)/(a*b) => c = a*b/(a+b)
        # c must be integer
        ab = a * b
        s = a + b
        if ab % s != 0:
            return None
        c = ab // s
        if c < 5 or c > 10000000:
            return None
        # Check c is not already in result (excluding i, j)
        for k in range(len(result)):
            if k != i and k != j and result[k] == c:
                return None
        return c
    
    def local_optimize(result, deadline):
        """Try to reduce count by merging pairs."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(result)
            for i in range(n):
                if time.time() > deadline:
                    return result
                for j in range(i + 1, n):
                    c = try_merge_pair(result, i, j)
                    if c is not None:
                        new_result = result[:i] + result[i+1:j] + result[j+1:] + [c]
                        new_result.sort()
                        # Verify sum is still 1
                        s = sum(Fraction(1, d) for d in new_result)
                        if s == 1 and len(set(new_result)) == len(new_result):
                            result = new_result
                            changed = True
                            break
                if changed:
                    break
        return result
    
    # Phase 1: Greedy
    target = Fraction(1, 1)
    result = greedy_egyptian(target, min_den=5, max_den=10000000)
    
    if result is None:
        # Fallback: try with slightly different starting point
        for start_d in range(5, 20):
            if time.time() > deadline:
                break
            # Start with a fixed denominator, then greedy on remainder
            rem = target - Fraction(1, start_d)
            if rem > 0:
                rest = greedy_egyptian(rem, min_den=start_d + 1, max_den=10000000)
                if rest is not None:
                    candidate = [start_d] + rest
                    s = sum(Fraction(1, d) for d in candidate)
                    if s == 1 and len(set(candidate)) == len(candidate) and len(candidate) <= 256:
                        result = candidate
                        break
    
    if result is None:
        # DP fallback
        target_lcm = 2520
        reached = {0: []}
        for d in range(5, target_lcm + 1):
            if target_lcm % d:
                continue
            w = target_lcm // d
            for total, values in list(reached.items()):
                if total + w <= target_lcm and total + w not in reached:
                    reached[total + w] = values + [d]
            if target_lcm in reached:
                result = reached[target_lcm]
                break
    
    if result is None:
        raise RuntimeError("no baseline representation")
    
    # Phase 2: Local optimization to reduce count
    result = local_optimize(result, deadline)
    
    # Final validation
    s = sum(Fraction(1, d) for d in result)
    if s != 1 or len(set(result)) != len(result) or len(result) < 1 or len(result) > 256:
        # Return original greedy result if optimization broke it
        result = greedy_egyptian(Fraction(1, 1), min_den=5, max_den=10000000)
        if result is None:
            raise RuntimeError("no valid representation")
    
    return result
# EVOLVE-BLOCK-END
