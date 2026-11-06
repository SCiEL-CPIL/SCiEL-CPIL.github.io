import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def time_ok():
        return time.time() < deadline

    def is_valid(denoms):
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == 1

    # Search for k=9
    # We want d1 < d2 < ... < d9, d1 >= 5
    # Sum 1/di = 1
    
    def search(k, target, min_d, current_denoms):
        if not time_ok():
            return None
        
        if k == 0:
            if target == 0:
                return current_denoms
            return None
        
        if target <= 0:
            return None
            
        # Pruning 1: Max possible sum with k terms starting from min_d
        # The largest sum is achieved by using the smallest possible denominators: min_d, min_d+1, ..., min_d+k-1
        # If sum(1/(min_d+i) for i in range(k)) < target, it's impossible
        # Actually, we can just check if k * (1/min_d) < target, which is a weaker bound but faster.
        # Let's use the tighter bound: sum of reciprocals of the k smallest available denominators.
        
        # However, calculating the sum of k reciprocals every step is expensive.
        # Approximation: k * (1/min_d) is an upper bound on the sum? No, 1/min_d is the largest term.
        # The sum of k distinct reciprocals starting from min_d is < k/min_d.
        # If k/min_d < target, prune.
        if Fraction(k, min_d) < target:
            return None
            
        # Lower bound on the sum of k distinct reciprocals starting from min_d:
        # sum_{i=0}^{k-1} 1/(min_d+i)
        # If this sum < target, prune.
        # This is expensive to compute exactly with Fraction for every step.
        # We can use a float approximation for pruning, then verify with Fraction.
        
        min_sum_float = 0.0
        for i in range(k):
            min_sum_float += 1.0 / (min_d + i)
            
        if min_sum_float < float(target) * 0.999999: # Small epsilon for float errors
            return None

        # Determine the range for the next denominator d
        # 1/d <= target => d >= 1/target
        # d >= min_d
        d_min = max(min_d, int(1 / float(target)) + 1)
        
        # Upper bound for d:
        # We need to leave enough sum for the remaining k-1 terms.
        # The remaining sum is target - 1/d.
        # The maximum sum for k-1 terms starting from d+1 is (k-1)/(d+1).
        # So target - 1/d <= (k-1)/(d+1)
        # This inequality helps bound d from above?
        # Actually, if d is too large, 1/d is small, and we have to make up the difference with k-1 terms.
        # The smallest terms we can use next are d+1, d+2...
        # So target - 1/d must be >= sum_{i=1}^{k-1} 1/(d+i)
        # This is a complex condition.
        
        # Simpler upper bound:
        # The next term 1/d must be at least 1/(min_d + k - 1) ? No.
        # Let's just iterate d upwards and rely on the recursive call's pruning.
        # But we need an upper bound to avoid infinite loop.
        # If d is very large, 1/d is very small.
        # The remaining sum is approx target.
        # The max sum of k-1 terms starting from d+1 is (k-1)/(d+1).
        # So we need target - 1/d <= (k-1)/(d+1).
        # target <= 1/d + (k-1)/(d+1)
        # For large d, RHS ~ k/d. So d ~ k/target.
        # So d_max can be roughly int(k / float(target)) + 10.
        
        d_max = int(k / float(target)) + 10
        if d_max > 10000000:
            d_max = 10000000
            
        for d in range(d_min, d_max + 1):
            if not time_ok():
                return None
                
            new_target = target - Fraction(1, d)
            if new_target < 0:
                continue
                
            result = search(k - 1, new_target, d + 1, current_denoms + [d])
            if result is not None:
                return result
                
        return None

    # Start search for k=9
    # d1 must be at least 5.
    # We can iterate d1 from 5 to some limit.
    # If d1 is too large, the sum of 9 terms will be too small.
    # Min sum for d1=5 is ~1.094.
    # If d1=6, min sum is 1/6+...+1/14 ~ 0.99.
    # So d1 can be 5 or 6.
    
    for d1 in range(5, 7):
        if not time_ok():
            break
        t1 = 1 - Fraction(1, d1)
        res = search(8, t1, d1 + 1, [d1])
        if res is not None:
            if is_valid(res):
                return res

    # If k=9 not found, try k=10
    # We can use a similar search, but it will be much slower.
    # Instead, let's use the base strategy from the original program but with more bases.
    # Or just return a known k=10 solution to ensure score 1.0.
    
    # Known k=10 solution from greedy or previous search:
    # Let's try to find one quickly with a greedy-like approach or a small search.
    
    def find_k10():
        # Try bases of size 7
        # We want 3 more terms.
        # Base sum S. Remainder R = 1 - S.
        # Need 1/a + 1/b + 1/c = R.
        
        bases = [
            [5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 12],
            [5, 6, 7, 8, 9, 11, 12],
            [5, 6, 7, 8, 10, 11, 12],
            [5, 6, 7, 9, 10, 11, 12],
            [5, 6, 8, 9, 10, 11, 12],
            [5, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12],
            [5, 6, 7, 8, 9, 10, 13],
            [5, 6, 7, 8, 9, 11, 13],
        ]
        
        for base in bases:
            if not time_ok():
                break
            s = sum(Fraction(1, d) for d in base)
            r = 1 - s
            if r <= 0:
                continue
                
            d_min = max(base[-1] + 1, int(1 / float(r)) + 1)
            d_max = int(3 / float(r)) + 10
            
            if d_max > 10000000:
                d_max = 10000000
                
            for a in range(d_min, d_max + 1):
                if not time_ok():
                    break
                r2 = r - Fraction(1, a)
                if r2 <= 0:
                    continue
                    
                # Need 1/b + 1/c = r2
                b_min = max(a + 1, int(1 / float(r2)) + 1)
                b_max = int(2 / float(r2)) + 10
                
                if b_max > 10000000:
                    b_max = 10000000
                    
                for b in range(b_min, b_max + 1):
                    if not time_ok():
                        break
                    r3 = r2 - Fraction(1, b)
                    if r3 <= 0:
                        continue
                    c = 1 / float(r3)
                    c_int = round(c)
                    if c_int > b and c_int <= 10000000:
                        candidate = base + [a, b, c_int]
                        if is_valid(candidate):
                            return candidate
        return None

    res10 = find_k10()
    if res10:
        return res10

    # Fallback: Greedy
    R = Fraction(1, 1)
    d_prev = 4
    denoms = []
    while R > 0 and time_ok():
        d_min = max(d_prev + 1, int(1 / float(R)) + 1)
        d = d_min
        denoms.append(d)
        R -= Fraction(1, d)
        d_prev = d
        
    if is_valid(denoms):
        return denoms
        
    raise RuntimeError("No valid solution found")
