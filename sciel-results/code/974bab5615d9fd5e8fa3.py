"""
Improved program for Egyptian fraction of 5/31; minimizes largest denominator.
Combines DFS search with exact 2-term Diophantine solver.
"""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def find_two_term_completions(r, min_den, max_den):
        """
        Find all pairs (b, c) with min_den <= b < c <= max_den such that 1/b + 1/c = r.
        Uses the factorization method: (nb-d)(nc-d) = d^2 for r = n/d.
        Returns a list of (b, c) tuples.
        """
        if r <= 0:
            return []
        
        n = r.numerator
        d = r.denominator
        
        # Equation: (n*b - d) * (n*c - d) = d^2
        # Let u = n*b - d, v = n*c - d. Then u*v = d^2.
        # b = (u + d) / n, c = (v + d) / n.
        # We need u, v to be positive integers (since b, c > 0 and n, d > 0).
        # Also b < c implies u < v (since n > 0).
        # And b >= min_den implies u + d >= n * min_den => u >= n * min_den - d.
        
        d2 = d * d
        limit = isqrt(d2)
        
        results = []
        
        # Iterate over factors u of d^2
        # Optimization: Only check factors up to sqrt(d^2)
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            
            v = d2 // u
            
            # We require u < v to ensure b < c. If u == v, b == c, which is invalid for distinct denominators.
            if u >= v:
                continue
                
            # Check divisibility
            bu_num = u + d
            bv_num = v + d
            
            if bu_num % n != 0 or bv_num % n != 0:
                continue
                
            b = bu_num // n
            c = bv_num // n
            
            # Check constraints
            if b < min_den or c > max_den:
                continue
            if b >= c: # Distinctness and ordering
                continue
            if b <= 0 or c <= 0:
                continue
                
            results.append((b, c))
            
        return results

    def dfs(r, min_den, max_den, current, depth, max_depth=4):
        """
        DFS to find the first 'max_depth' denominators.
        At depth == max_depth, we try to complete with 2 terms exactly.
        """
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return False
            
        if r == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
            return True
            
        if r < 0:
            return False
            
        if depth >= max_depth:
            # Try to complete the remaining sum r with exactly 2 terms
            # Note: The problem allows 1..256 terms. If we reach depth 4, we can still use more terms,
            # but solving for 2 terms is a strong heuristic. If 2 terms fail, we could recurse deeper,
            # but that explodes search space. Given 5/31 is small, 3 or 4 terms often suffice for optimal max_den.
            # Let's allow one more level of depth if 2-term fails? No, keep it simple first.
            # Actually, let's just try 2-term completion here.
            completions = find_two_term_completions(r, min_den, max_den)
            for b, c in completions:
                candidate = current + [b, c]
                m = max(candidate)
                if m < best_max:
                    best_max = m
                    best_solution = candidate
            return False # We didn't find a single path that "ends" the recursion in the standard way, but we updated best.
            
        # Pruning:
        # 1. If we use the smallest possible denominator d_start, the remaining value must be representable.
        # 2. Upper bound on sum of remaining terms.
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
            
        # If even using the largest possible denominators for the remaining slots we can't reach the target sum?
        # Max sum with (max_depth - depth) terms starting from d_start is approx (max_depth - depth) / d_start.
        # If r > that, we can't complete.
        remaining_terms = max_depth - depth
        if remaining_terms <= 0:
            return False
        # Loose upper bound: r <= remaining_terms / d_start
        # More precise: r <= sum_{i=0}^{remaining_terms-1} 1/(d_start + i)
        # For now, use the loose bound for speed, it's safe.
        if r > Fraction(remaining_terms, d_start):
            return False
            
        # Iterate d
        # Optimization: We don't need to go all the way to max_den.
        # If 1/d + (remaining_terms - 1)/d > r, then d is too small? No, 1/d is the largest term.
        # We need 1/d <= r => d >= 1/r. This is d_start.
        # Also, if we pick d, the remainder is r - 1/d.
        # The next min_den is d+1.
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
                
            new_r = r - Fraction(1, d)
            if new_r < 0:
                # Since d is increasing, new_r will only get more negative.
                break
                
            current.append(d)
            # Recurse
            dfs(new_r, d + 1, max_den, current, depth + 1, max_depth)
            current.pop()
            
            # Pruning: If we found a solution, we can stop early if we are just looking for ANY solution?
            # No, we want the MINIMUM max_den. So we must continue searching unless we are sure no better solution exists.
            # However, if we found a solution with max_den = M, any branch where d >= M cannot improve it (since max(current+...) >= d).
            # Actually, if d >= best_max, we can break the loop?
            # If d >= best_max, then max(candidate) >= d >= best_max. So we cannot improve best_max.
            if d >= best_max:
                break

    # Initial Phase: Greedy to get a decent upper bound quickly
    # 5/31 approx 0.161. 1/7 approx 0.142.
    # Try a few greedy starts to set best_max.
    def greedy_complete(r, min_den, max_den, max_terms=256):
        values = []
        for _ in range(max_terms):
            if time.time() > deadline:
                return None
            if r == 0:
                return values
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None

    # Try greedy with different first denominators to establish a baseline
    for d1 in range(7, 20):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        sol = greedy_complete(r1, d1 + 1, 10000)
        if sol:
            full_sol = [d1] + sol
            if max(full_sol) < best_max:
                best_max = max(full_sol)
                best_solution = full_sol

    # Main DFS Search
    # We search for the first 3 denominators, then solve the rest with 2 terms.
    # Why 3? 5/31 is small. 1/7 + 1/23 + ...
    # Let's try depth 3 first. If it takes too long, we might need to adjust.
    # The 2-term solver is very fast.
    
    # Set a reasonable max_den for the search. 
    # If we don't have a good initial bound, use a high one.
    # But we want to MINIMIZE the largest denominator.
    # The DFS iterates d from d_start upwards.
    # If we find a solution with max M, we can restrict future searches to d < M?
    # Actually, the loop `if d >= best_max: break` handles this dynamically.
    
    # Start DFS
    # min_den for first term is 1? No, 1/1 = 1 > 5/31. So d1 >= 7.
    # The formula d_start = ceil(1/r) handles this.
    dfs(target, 1, 10000000, [], 0, max_depth=3)
    
    # If the 3-depth + 2-term search didn't find a solution (unlikely), or if we have time, try depth 2 + 3-term?
    # Or just depth 4 + 2-term?
    # Let's try depth 4 if we have time and no solution found yet or if we want to be thorough.
    # But depth 3 is likely sufficient for 5/31.
    
    # Let's do a second pass with depth 2, but using a 3-term solver?
    # 3-term solver is harder. 
    # Alternative: Depth 4 + 2-term solver.
    if time.time() < deadline - 10:
        # Reset best if we want to explore deeper? No, keep the best found.
        # But the DFS with depth 3 might have pruned branches where d1 was large but led to a better total max?
        # No, the pruning `if d >= best_max: break` is safe.
        # However, if the optimal solution has max_den = 100, and greedy found 1000,
        # the DFS will search d1 up to 99.
        
        # Let's try depth 4 just in case.
        # To avoid re-doing work, we can just run it. It will prune based on current best_max.
        dfs(target, 1, 10000000, [], 0, max_depth=4)

    if best_solution is not None:
        # Verify
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution) and len(best_solution) <= 256:
            return best_solution

    # Fallback: Simple greedy
    r = target
    values = []
    min_den = 1
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < min_den:
            d = min_den
        values.append(d)
        r -= Fraction(1, d)
        min_den = d + 1
    return values

# EVOLVE-BLOCK-END
