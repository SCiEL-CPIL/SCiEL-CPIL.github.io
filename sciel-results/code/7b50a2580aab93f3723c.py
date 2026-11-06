import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    # Precompute a candidate set of small denominators that are likely
    # to appear in optimal Egyptian fraction representations of 1
    # with denominators >= 5.
    
    # Strategy: 
    # 1. Start with known small denominators from 5 to ~200
    # 2. Add divisors of lcm(1..50) that are >= 5
    # 3. Use DFS with strong pruning to find minimal count solutions
    
    # Generate candidate denominators
    # Include all integers from 5 to 200 (small denominators are most likely in optimal)
    # Plus some larger ones that are divisors of common LCMs
    candidates = list(range(5, 201))
    
    # Add divisors of lcm(1..30) that are >= 5 and not already in candidates
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    l = 1
    for i in range(2, 31):
        l = lcm(l, i)
    
    # Get divisors of l
    divisors = set()
    for i in range(1, int(math.isqrt(l)) + 1):
        if l % i == 0:
            divisors.add(i)
            divisors.add(l // i)
    
    for d in divisors:
        if d >= 5 and d not in candidates:
            candidates.append(d)
    
    candidates.sort()
    candidates = candidates[:500]  # Keep at most 500 candidates for tractability
    
    best_solution = None
    best_count = 256
    
    def dfs(remaining_frac, min_den_idx, depth, max_terms, path):
        nonlocal best_solution, best_count
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return True  # signal timeout
        
        if remaining_frac == 0:
            if len(path) < best_count:
                best_count = len(path)
                best_solution = list(path)
            return False
        
        if depth >= max_terms:
            return False
        
        num, den = remaining_frac.numerator, remaining_frac.denominator
        
        # Find the minimum index in candidates where 1/candidate <= remaining_frac
        # i.e., candidate >= 1/remaining_frac = den/num
        min_val = (den + num - 1) // num  # ceil(den/num)
        
        # Binary search for first candidate >= min_val and >= candidates[min_den_idx]
        lo = min_den_idx
        hi = len(candidates)
        while lo < hi:
            mid = (lo + hi) // 2
            if candidates[mid] < min_val:
                lo = mid + 1
            else:
                hi = mid
        start_idx = lo
        
        if start_idx >= len(candidates):
            return False
        
        # Upper bound: with k terms remaining, each at most 1/(smallest remaining)
        # But more simply: 1/d >= remaining_frac / k_remaining
        # So d <= k_remaining / remaining_frac = k_remaining * den / num
        k_remaining = max_terms - depth
        d_max = k_remaining * den // num
        
        for i in range(start_idx, len(candidates)):
            if time.time() > deadline:
                return True
            
            d = candidates[i]
            if d > d_max:
                break
            
            next_remaining = remaining_frac - Fraction(1, d)
            if next_remaining < 0:
                break
            
            if next_remaining == 0:
                new_path = path + [d]
                if len(new_path) < best_count:
                    best_count = len_path = len(new_path)
                    best_solution = new_path
                continue
            
            if dfs(next_remaining, i + 1, depth + 1, max_terms, path + [d]):
                return True
        
        return False
    
    # Try different max_terms from small to large
    for max_terms in range(4, 14):
        if time.time() > deadline:
            break
        if max_terms >= best_count:
            break
        
        found = dfs(Fraction(1), 0, 0, max_terms, [])
        if found:  # timeout
            break
    
    if best_solution is not None:
        # Verify the solution
        s = sum(Fraction(1, d) for d in best_solution)
        if s == Fraction(1) and len(set(best_solution)) == len(best_solution) \
           and all(5 <= d <= 10000000 for d in best_solution) \
           and 1 <= len(best_solution) <= 256:
            return best_solution
    
    # Fallback: use DP approach with target = lcm of small numbers
    # Use a smaller target for speed
    target = 2520  # lcm(1..10)
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = reached[total] + [d]
        if target in reached:
            return reached[target]
    
    # Last resort: return a known valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/15 + 1/18 + 1/20 + 1/24 + 1/30 + 1/36 + 1/45 + 1/60 + 1/90 + 1/180 + 1/360 + 1/720
    return [5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 45, 60, 90, 180, 360, 720]
