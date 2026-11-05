"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill

    def check_time():
        return time.time() > deadline

    # Greedy algorithm: repeatedly pick the largest denominator d >= min_den
    # such that 1/d <= remaining fraction
    def greedy_egyptian(target_num, target_den, min_den=5, max_terms=10):
        result = []
        num, den = target_num, target_den
        for _ in range(max_terms):
            if num == 0:
                return result
            if num * min_den > den * (max_terms - len(result)):
                return None  # can't fit in remaining terms
            
            # Find smallest d >= min_den such that 1/d <= num/den
            # i.e., d >= den/num
            d_low = max(min_den, (den + num - 1) // num)
            
            # Try d_low, d_low+1, ... until we find one that works
            # For greedy, we want the LARGEST 1/d that is <= num/den
            # That means the SMALLEST d >= den/num
            d = d_low
            
            # Verify: 1/d <= num/den means den <= num*d
            if num * d < den:
                d += 1
            
            # Check if this denominator is valid (distinct, in range)
            if d in result or d > 10000000:
                # Try next few values
                for d_try in range(d + 1, d + 1000):
                    if d_try not in result and d_try <= 10000000:
                        d = d_try
                        break
                else:
                    return None
            
            result.append(d)
            
            # Update: remaining = num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            # Reduce fraction
            if new_num == 0:
                return result
            
            g = math.gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
            
            min_den = d + 1  # next denominator must be larger
            
            if check_time():
                return None
        
        if num == 0:
            return result
        return None

    # First, get a quick valid solution using greedy
    best_result = greedy_egyptian(1, 1, 5, 10)
    
    if best_result is None:
        # Try with more terms
        best_result = greedy_egyptian(1, 1, 5, 15)
    
    if best_result is None:
        # Fallback: known small representation
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
        # Let's use a simpler known one or construct via DP
        # Actually, let's try a different greedy approach
        best_result = None
        for max_t in range(6, 20):
            best_result = greedy_egyptian(1, 1, 5, max_t)
            if best_result is not None:
                break
    
    if best_result is None:
        # Last resort: use DP with small denominators
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best_result = reached[target]
                break
    
    if best_result is None:
        raise RuntimeError("no representation found")

    best_count = len(best_result)

    # Now try to find a shorter representation using DFS
    def dfs(remaining_num, remaining_den, min_den, current, depth, max_terms):
        nonlocal best_result, best_count
        
        if check_time():
            return False
        
        if remaining_num == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_result = list(current)
            return True
        
        if remaining_num <= 0 or depth + 1 > max_terms:
            return False
        
        # Pruning: max possible sum with remaining terms
        if remaining_num * min_den > remaining_den * (max_terms - depth):
            return False
        
        d_low = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        K_rem = max_terms - depth - 1
        if K_rem < 0:
            return False
            
        # Estimate d_high
        d_high = int(K_rem * remaining_den / remaining_num) + 100
        d_high = min(d_high, 10000000)
        
        if d_high < d_low:
            return False
        
        # Limit the range to avoid huge iterations
        if d_high - d_low > 100000:
            d_high = d_low + 100000
        
        for d in range(d_low, d_high + 1):
            if check_time():
                return False
            
            # Check if 1/d <= remaining
            if d * remaining_num < remaining_den:
                continue
            
            # Check distinctness
            if d in current:
                continue
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num == 0:
                current.append(d)
                if len(current) < best_count:
                    best_count = len(current)
                    best_result = list(current)
                current.pop()
                return True
            
            current.append(d)
            dfs(new_num, new_den, d + 1, current, depth + 1, max_terms)
            current.pop()
            
            if best_count <= depth + 1:
                return True
        
        return False

    # Try to improve with shorter sequences
    # Start from best_count - 1 and go down to 4
    for k in range(best_count - 1, 3, -1):
        if check_time():
            break
        if dfs(1, 1, 5, [], 0, k):
            if best_count == k:
                break

    return best_result

# EVOLVE-BLOCK-END
