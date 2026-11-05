"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find 1..256 distinct denominators in [1,10000000] summing reciprocally to 5/31,
    minimizing the largest denominator.
    
    Strategy: Binary search on the maximum denominator M. For each M, use DFS
    backtracking to check if a subset of {1,...,M} sums reciprocally to 5/31.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    def dfs(idx, remaining, chosen, M):
        """
        Try to form 'remaining' using a subset of {idx, idx+1, ..., M}.
        Returns the list of chosen denominators if successful, else None.
        """
        if time.time() > deadline:
            return None
        
        if remaining == 0:
            return list(chosen)
        
        if remaining < 0:
            return None
        
        if idx > M:
            return None
        
        # Pruning: even if we take all remaining denominators, can we reach 'remaining'?
        # Max possible sum from idx to M: sum(1/i for i in range(idx, M+1))
        # This is expensive to compute each time, so use a bound.
        # sum(1/i for i=idx to M) <= (M - idx + 1) / idx
        max_possible = Fraction(M - idx + 1, idx)
        if remaining > max_possible:
            return None
        
        # Pruning: if 1/idx alone exceeds remaining, we can't use idx or any larger
        # (since all subsequent are smaller reciprocals... wait no, 1/idx > 1/(idx+1) > ...)
        # Actually 1/idx is the LARGEST remaining reciprocal. If 1/idx > remaining,
        # we still might use it if remaining >= 1/idx... no wait.
        # If 1/idx > remaining, we CANNOT use idx (would overshoot), and since all
        # subsequent 1/j for j > idx are smaller, we might still use them.
        # But if even the sum of all remaining can't reach 'remaining', we prune (done above).
        
        # Option 1: include idx
        if Fraction(1, idx) <= remaining:
            result = dfs(idx + 1, remaining - Fraction(1, idx), chosen + [idx], M)
            if result is not None:
                return result
        
        # Option 2: exclude idx
        result = dfs(idx + 1, remaining, chosen, M)
        if result is not None:
            return result
        
        return None
    
    # Binary search on M
    # Lower bound: we need at least some denominators. 1/7 ≈ 0.143, 5/31 ≈ 0.161
    # So we need at least denominators that can sum to 5/31.
    # Minimum M: try from small values up.
    
    # First, try small M values directly
    # M must be at least 7 (since 1/7 is needed as a large term)
    # Let's try M from 7 upward
    
    # But DFS over 2^M is too slow for large M. Let's be smart.
    # For small M (up to ~20), DFS with pruning should work.
    # For larger M, we need a different strategy.
    
    # Let's first try the 3-term approach to get a good upper bound
    best_max = float('inf')
    best_solution = None
    
    # 3-term search (fast, gives upper bound)
    for a in range(7, 19):
        if time.time() > deadline:
            break
        num = 5 * a - 31
        den = 31 * a
        if num <= 0:
            continue
        
        b_lower = max(a + 1, den // num + 1)
        b_upper = (2 * den) // num
        
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            numerator_c = den * b
            denominator_c = num * b - den
            if denominator_c <= 0:
                continue
            if numerator_c % denominator_c != 0:
                continue
            c = numerator_c // denominator_c
            if c <= b or c > 10000000:
                continue
            lhs = 31 * (b * c + a * c + a * b)
            rhs = 5 * a * b * c
            if lhs != rhs:
                continue
            current_max = max(a, b, c)
            if current_max < best_max:
                best_max = current_max
                best_solution = [a, b, c]
    
    # Now try to improve using DFS with M < best_max
    # Try M from 7 to best_max - 1
    for M in range(7, min(best_max, 50)):  # cap at 50 for DFS feasibility
        if time.time() > deadline:
            break
        
        # Quick feasibility check: sum of 1/i for i=1 to M must be >= target
        # And we need a subset summing to exactly target
        # Use DFS but with aggressive pruning
        
        # Precompute suffix sums for pruning
        # suffix_sum[i] = sum(1/j for j in range(i, M+1))
        suffix_sum = [Fraction(0)] * (M + 2)
        for i in range(M, 0, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, i)
        
        def dfs2(idx, remaining, chosen):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                return list(chosen)
            
            if remaining < 0:
                return None
            
            if idx > M:
                return None
            
            # Pruning: can't reach remaining with what's left
            if remaining > suffix_sum[idx]:
                return None
            
            # Pruning: if 1/idx > remaining, skip idx (can't include it)
            # But we can still try excluding it
            if Fraction(1, idx) > remaining:
                # Can't include idx, must exclude
                return dfs2(idx + 1, remaining, chosen)
            
            # Option 1: include idx
            result = dfs2(idx + 1, remaining - Fraction(1, idx), chosen + [idx])
            if result is not None:
                return result
            
            # Option 2: exclude idx
            result = dfs2(idx + 1, remaining, chosen)
            if result is not None:
                return result
            
            return None
        
        solution = dfs2(1, target, [])
        if solution is not None:
            # Verify
            s = sum(Fraction(1, x) for x in solution)
            if s == target and len(solution) <= 256 and len(set(solution)) == len(solution):
                if max(solution) <= M:
                    return sorted(solution)
    
    # If DFS didn't find anything better, return the 3-term solution
    if best_solution is not None:
        return best_solution
    else:
        return [8, 31, 248]

# EVOLVE-BLOCK-END
