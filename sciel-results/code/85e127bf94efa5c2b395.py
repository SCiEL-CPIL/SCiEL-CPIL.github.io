import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    target = Fraction(5, 31)
    
    # Known good solution
    best_solution = [8, 31, 248]
    best_max = 248

    def _verify(denoms):
        if not denoms:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target

    def _lcm(a, b):
        return a * b // math.gcd(a, b)

    def _lcm_range(n):
        result = 1
        for i in range(2, n + 1):
            result = _lcm(result, i)
            if result > 10**18:  # Too large for practical use
                return None
        return result

    # Strategy: For each M from best_max-1 down, try to find a subset of {1..M}
    # whose reciprocals sum to 5/31.
    # Use meet-in-the-middle for moderate M.
    
    # First, let's try a direct search for small M using DFS with strong pruning.
    # But also try the LCM-based subset sum approach.

    # Approach 1: DFS with integer arithmetic and aggressive pruning
    def _dfs_search(M, max_terms=8):
        """Search for Egyptian fraction of 5/31 with all denominators <= M."""
        if time.time() > deadline - 1:
            return None
        
        # Try different numbers of terms
        for k in range(2, max_terms + 1):
            if time.time() > deadline - 1:
                return None
            result = _dfs_k(5, 31, 0, M, k, set(), [])
            if result is not None:
                return result
        return None

    def _dfs_k(num, den, last_d, M, k, used, path):
        if time.time() > deadline - 1:
            return None
        if k == 0:
            return list(path) if num == 0 else None
        if num <= 0:
            return None
        
        # Lower bound: 1/d < num/den => d > den/num
        d_min = den // num + 1
        if d_min <= last_d:
            d_min = last_d + 1
        if d_min > M:
            return None
        
        # Pruning: max possible sum of k terms from [d_min, M]
        # Max sum = sum of 1/d for d = d_min, d_min+1, ..., d_min+k-1 (if all <= M)
        max_possible = 0.0
        for i in range(k):
            d = d_min + i
            if d > M:
                break
            max_possible += 1.0 / d
        if max_possible < num / den - 1e-12:
            return None
        
        # Min possible sum = sum of 1/d for d = M, M-1, ..., M-k+1 (if all >= d_min)
        min_possible = 0.0
        for i in range(k):
            d = M - i
            if d < d_min:
                break
            min_possible += 1.0 / d
        if min_possible > num / den + 1e-12:
            return None
        
        for d in range(d_min, M + 1):
            if time.time() > deadline - 1:
                return None
            if d in used:
                continue
            
            new_num = num * d - den
            new_den = den * d
            if new_num < 0:
                break
            
            g = math.gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            used.add(d)
            path.append(d)
            res = _dfs_k(new_num, new_den, d, M, k - 1, used, path)
            path.pop()
            used.discard(d)
            
            if res is not None:
                return res
        return None

    # Approach 2: Meet-in-the-middle using LCM
    def _mitm_search(M, max_terms=6):
        """Meet-in-the-middle: split denominators into two halves."""
        if time.time() > deadline - 1:
            return None
        
        L = _lcm_range(M)
        if L is None:
            return None
        
        target_num = 5 * L // 31
        if 5 * L % 31 != 0:
            return None
        
        # Each denominator d contributes weight L/d
        # We need subset of {L/1, L/2, ..., L/M} summing to target_num
        # with distinct denominators (weights are distinct since L/d are distinct for distinct d)
        
        # For MITM, split into first half and second half
        mid = M // 2
        first_half = list(range(1, mid + 1))
        second_half = list(range(mid + 1, M + 1))
        
        # Generate all subset sums for each half
        # But this is 2^mid which is too large for mid > 20
        # So only use MITM for small M
        if M > 25:
            return None
        
        # For very small M, brute force all subsets
        all_denoms = list(range(1, M + 1))
        for k in range(2, min(max_terms, M) + 1):
            if time.time() > deadline - 1:
                return None
            for combo in combinations(all_denoms, k):
                if time.time() > deadline - 1:
                    return None
                # Check if sum of 1/d for d in combo equals 5/31
                s = sum(Fraction(1, d) for d in combo)
                if s == target:
                    return list(combo)
        return None

    # Search M from best_max - 1 down to 10
    for M in range(best_max - 1, 9, -1):
        if time.time() > deadline - 3:
            break
        
        found = None
        
        # For small M, try brute force / MITM
        if M <= 25:
            found = _mitm_search(M, max_terms=min(8, M))
        
        # For larger M, try DFS
        if found is None and M <= 100:
            found = _dfs_search(M, max_terms=6)
        
        if found is not None:
            if _verify(found):
                actual_max = max(found)
                if actual_max < best_max:
                    best_max = actual_max
                    best_solution = found
        
        # Early termination if we've reached a very small M
        if best_max <= 15:
            break

    return best_solution
