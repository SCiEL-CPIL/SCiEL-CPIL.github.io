"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    def verify(denoms):
        """Verify that denoms are distinct, in [5, 10000000], and sum of reciprocals is 1."""
        if not denoms:
            return False
        if len(denoms) != len(set(denoms)):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1

    def dfs_search(num, den, last_d, remaining_depth, path, max_branch=100):
        """
        DFS with pruning, trying smallest denominators first.
        """
        if time.time() > deadline:
            return None
        
        if num == 0:
            return path if remaining_depth >= 0 else None
        
        if remaining_depth == 0:
            return None
        
        # Minimum denominator
        min_d = max(last_d + 1, (den + num - 1) // num)
        if min_d < 5:
            min_d = 5
        
        # Maximum denominator for feasibility
        # We need 1/d <= num/den, and we have remaining_depth terms left
        # The largest possible sum with remaining_depth terms starting from min_d+1
        # is sum of 1/(min_d+1) + 1/(min_d+2) + ... 
        # For a quick check: if 1/min_d * remaining_depth < num/den, infeasible
        if min_d * num > den * remaining_depth + 1:
            return None
        
        max_d = min(10000000, (den * remaining_depth) // num + 1)
        if max_d < min_d:
            return None
        
        # Try denominators from min_d upward, limited branching
        count = 0
        for d in range(min_d, min_d + max_branch):
            if d > max_d or d > 10000000:
                break
            if d < 5:
                continue
            
            # 1/d must be <= num/den
            if d * num < den:
                continue
            
            new_num = num * d - den
            if new_num < 0:
                break
            
            if new_num == 0:
                if remaining_depth == 1:
                    return path + [d]
                else:
                    continue
            
            new_den = den * d
            g = math.gcd(new_num, new_den)
            rn = new_num // g
            rd = new_den // g
            
            # Pruning: check if remaining sum can be achieved with remaining_depth-1 terms
            # Each term is at most 1/(d+1), so max additional sum = remaining_depth-1 terms * 1/(d+1)
            # Actually more precisely: sum of 1/(d+1) + 1/(d+2) + ... 
            # Quick bound: (remaining_depth - 1) / (d + 1) >= rn/rd
            if (remaining_depth - 1) * rd < rn * (d + 1) + 1:
                # Can't achieve the remaining sum
                continue
            
            count += 1
            if count > max_branch:
                break
            
            result = dfs_search(rn, rd, d, remaining_depth - 1, path + [d], max_branch)
            if result is not None:
                return result
        
        return None

    def greedy_construct():
        """Greedy Egyptian fraction with denominators >= 5."""
        denoms = []
        target = Fraction(1, 1)
        used = set()
        min_d = 5
        
        while target > 0 and len(denoms) < 256:
            # Find smallest d >= min_d such that 1/d <= target
            d = max(min_d, math.ceil(1.0 / float(target)) + 1)
            if d < 5:
                d = 5
            while d in used or Fraction(1, d) > target:
                d += 1
            if d > 10000000:
                break
            denoms.append(d)
            used.add(d)
            min_d = d + 1
            target -= Fraction(1, d)
        
        if target == 0 and len(denoms) <= 256 and verify(denoms):
            return denoms
        return None

    # Try DFS first with increasing depth limits
    for depth in range(5, 16):
        if time.time() > deadline - 10:
            break
        result = dfs_search(1, 1, 4, depth, [], max_branch=50)
        if result is not None and verify(result):
            return result
    
    # Fallback to greedy
    result = greedy_construct()
    if result is not None:
        return result
    
    # Absolute last resort: return a known valid solution if we can find one quickly
    # For now, return an empty list or raise an error
    return None
# EVOLVE-BLOCK-END
