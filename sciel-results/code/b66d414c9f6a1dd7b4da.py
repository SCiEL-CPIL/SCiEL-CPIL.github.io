"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _try_complete(r, used, max_d, start_d, depth=0, max_depth=5):
    """Try to complete r with at most max_depth more terms, all <= max_d, >= start_d, not in used."""
    if time.time() > _DEADLINE:
        return None
    if r == 0:
        return []
    if depth >= max_depth:
        return None
    if r < 0:
        return None
    
    # Lower bound: 1/d < r => d > 1/r
    # Also d >= start_d, d not in used
    # Upper bound: d <= max_d
    # We need sum of remaining reciprocals = r
    # If we pick d, then r - 1/d must be representable with (max_depth - depth - 1) terms
    
    # Minimum possible value of sum of k distinct reciprocals starting from d_min is achieved by largest denominators
    # But for feasibility, we need r <= sum of 1/(d) for the largest possible denominators? No, we need r to be at least the sum of the smallest possible reciprocals?
    # Actually, the maximum sum we can get with k terms starting from d_min is 1/d_min + 1/(d_min+1) + ...
    # The minimum sum is 1/max_d + 1/(max_d-1) + ... but denominators must be distinct and >= start_d.
    
    # Simple bound: if r > sum of 1/d for the k smallest available denominators, impossible.
    # k = max_depth - depth
    k = max_depth - depth
    # The smallest available denominators are the first k integers >= start_d not in used
    # For an upper bound on r, let's just check if r is not too large
    # Sum of 1/d for d in [start_d, start_d+k) is an upper bound on what we can achieve with k terms
    upper_sum = 0
    d_check = start_d
    count = 0
    while count < k:
        if d_check not in used:
            upper_sum += Fraction(1, d_check)
            count += 1
        d_check += 1
        if d_check > max_d:
            break
    if r > upper_sum:
        return None
    
    # Also, r must be at least the sum of the k largest possible reciprocals? No, that's the minimum.
    # The minimum sum with k terms <= max_d is approx k/max_d.
    # If r < k/max_d, it might be impossible, but let's not prune too aggressively.
    
    # Iterate possible d
    # d must satisfy 1/d < r => d > 1/r
    d_min = max(start_d, (r.denominator // r.numerator) + 1)
    if r.numerator * d_min <= r.denominator: # 1/d_min >= r
        d_min += 1
        
    for d in range(d_min, max_d + 1):
        if time.time() > _DEADLINE:
            return None
        if d in used:
            continue
        if Fraction(1, d) >= r:
            continue
        new_r = r - Fraction(1, d)
        if new_r < 0:
            continue
        used.add(d)
        res = _try_complete(new_r, used, max_d, d + 1, depth + 1, max_depth)
        used.discard(d)
        if res is not None:
            return [d] + res
    return None

_DEADLINE = 0.0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55.0  # Leave 5s buffer for safety
    
    target = Fraction(5, 31)
    
    # Known good solution: [8, 28, 1736]
    # Let's verify: 1/8 + 1/28 + 1/1736
    # 1/8 = 217/1736
    # 1/28 = 62/1736
    # 1/1736 = 1/1736
    # Sum = (217+62+1)/1736 = 280/1736 = 35/217 = 5/31. Correct.
    
    best_solution = [8, 28, 1736]
    best_max = 1736
    
    # Strategy: Search for solutions with smaller max denominator.
    # We will iterate through potential max_d values from small to large?
    # Or iterate through first denominators and try to find any solution, keeping the best.
    
    # The first denominator d1 must be < 31/5 = 6.2, so d1 <= 6?
    # Wait, 1/d1 < 5/31 => d1 > 31/5 = 6.2 => d1 >= 7.
    # So d1 is at least 7.
    
    # Let's try to find a solution with max_d < 1736.
    # We can do a systematic search over d1, d2, d3...
    # But the search space is large.
    
    # Let's try a randomized approach with local optimization?
    # Or a more directed search.
    
    # Since we have a known solution with max 1736, let's try to improve it.
    # We can try to replace the largest denominator 1736 with a combination of smaller ones?
    # Or find a completely different set.
    
    # Let's do a DFS for small term counts (2, 3, 4, 5) with increasing max_d bound.
    # If we find a solution with max_d < current best, update best.
    
    # Actually, let's just run a time-limited search for any valid solution and keep the best.
    # We'll use a recursive search with pruning.
    
    # To speed up, we can pre-calculate some bounds.
    
    # Let's try searching for solutions with max_d in a specific range?
    # No, we want to minimize max_d.
    
    # Let's iterate d1 from 7 upwards.
    # For each d1, iterate d2 from d1+1 upwards.
    # For each d1, d2, iterate d3...
    
    # This might be too slow if we go deep.
    # Let's limit the number of terms to 5 or 6.
    
    # We'll use the _try_complete function which does DFS.
    # We need to call it for different starting conditions.
    
    # Let's try to find a 3-term solution first?
    # 1/a + 1/b + 1/c = 5/31
    # If we fix a, then 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # Then (b+c)/(bc) = (5a-31)/(31a)
    # 31a(b+c) = (5a-31)bc
    # This is a Diophantine equation.
    
    # Let's just use the generic search.
    
    # To make it efficient, we'll try different "max_d" limits?
    # No, just find any solution and track the min max_d.
    
    # We'll run the search for a few seconds.
    
    # Random restarts might help?
    # Let's stick to systematic search for d1, d2, d3, d4.
    
    # d1: 7 to 20 (smaller d1 means larger 1/d1, leaving less for others)
    # If d1 is too large, 1/d1 is small, leaving a large remainder, requiring many terms or large denominators.
    # If d1 is small (7), 1/7 = 0.142, 5/31 = 0.161. Remainder = 0.019.
    # 1/d2 < 0.019 => d2 > 52.
    
    # Let's iterate d1 from 7 to 30.
    for d1 in range(7, 31):
        if time.time() > _DEADLINE - 2:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        # Now find d2, d3... such that sum = r1
        # d2 > 1/r1
        d2_min = (r1.denominator // r1.numerator) + 1
        if r1.numerator * d2_min <= r1.denominator:
            d2_min += 1
            
        # Limit d2 search range to avoid too many iterations
        # d2 should be such that remaining can be solved
        # Let's try d2 from d2_min to d2_min + 100
        for d2 in range(d2_min, d2_min + 150):
            if time.time() > _DEADLINE - 2:
                break
            if d2 == d1:
                continue
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            # Now find d3, d4... for r2
            # Use _try_complete with max_depth = 3 (d3, d4, d5)
            # But _try_complete returns the list of denominators.
            # We need to ensure all are distinct and > d2.
            # _try_complete handles distinctness via 'used' set and start_d.
            
            # We want to minimize the max denominator.
            # _try_complete doesn't know about 'best_max'.
            # We can pass best_max - 1 as the max_d limit to _try_complete?
            # If it finds a solution with max_d < best_max, we update.
            
            # However, if we limit max_d, we might miss solutions where an intermediate term is large but the last is small?
            # No, all terms must be <= max_d.
            
            # Let's call _try_complete with max_d = best_max - 1.
            # If it returns a solution, we update best_max.
            
            used = {d1, d2}
            res = _try_complete(r2, used, best_max - 1, d2 + 1, 0, 4) # Max 4 more terms
            if res is not None:
                sol = [d1, d2] + res
                # Verify distinctness and range
                if len(set(sol)) == len(sol) and all(1 <= d <= 10000000 for d in sol):
                    # Verify sum
                    s = sum(Fraction(1, d) for d in sol)
                    if s == target:
                        max_d = max(sol)
                        if max_d < best_max:
                            best_max = max_d
                            best_solution = sol
                            # Update deadline? No, keep searching for better.
                            # But we can now limit search to even smaller max_d?
                            # The next call to _try_complete will use best_max - 1 which is smaller.
    
    # Also try randomization to escape local minima?
    # If we have time left, do some random trials.
    if time.time() < _DEADLINE - 5:
        for _ in range(1000):
            if time.time() > _DEADLINE - 2:
                break
            # Pick random d1
            d1 = random.randint(7, 50)
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                continue
            used = {d1}
            # Try to complete with random start?
            # _try_complete is deterministic DFS.
            # Maybe just call it with a random max_d limit?
            # Or just call it with best_max - 1.
            res = _try_complete(r1, used, best_max - 1, d1 + 1, 0, 5)
            if res is not None:
                sol = [d1] + res
                if len(set(sol)) == len(sol):
                    s = sum(Fraction(1, d) for d in sol)
                    if s == target:
                        max_d = max(sol)
                        if max_d < best_max:
                            best_max = max_d
                            best_solution = sol

    return best_solution
# EVOLVE-BLOCK-END
