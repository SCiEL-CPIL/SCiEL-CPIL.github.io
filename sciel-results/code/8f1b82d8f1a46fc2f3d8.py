"""Rewritten program using a different approach: precomputed optimal Egyptian fraction representations."""
# EVOLVE-BLOCK-START
def construct():
    # Precomputed minimal-length Egyptian fraction representations of 1
    # with all denominators >= 5.
    #
    # Strategy: find small sets of distinct integers d_i >= 5 such that
    # sum(1/d_i) = 1, minimizing the number of terms.
    #
    # Known results for Egyptian fractions of 1:
    # - 2 terms: 1/2 + 1/2 (not distinct), 1/3+1/3 (not distinct), etc. No solution with distinct denominators >= 5.
    # - 3 terms: 1/a + 1/b + 1/c = 1 with a,b,c >= 5, distinct.
    #   Let's search: assume a < b < c, a >= 5.
    #   1/a + 1/b + 1/c = 1
    #   Try a=5: 1/5 + 1/b + 1/c = 1 => 1/b + 1/c = 4/5
    #     1/b <= 4/5 - 1/10000000 approx, but more precisely:
    #     1/b + 1/c = 4/5, b > 5.
    #     1/b < 4/5 => b > 5/4 = 1.25, so b >= 6 (but b > a=5).
    #     Also 1/b > 4/5 - 1/6 = 24/30 - 5/30 = 19/30 => b < 30/19 ≈ 1.57... wait that's wrong.
    #     Let me redo: 1/b + 1/c = 4/5, with b < c.
    #     2/b > 4/5 => b < 10/4 = 2.5. But b > a = 5. Contradiction. So no solution with a=5.
    #   Try a=6: 1/6 + 1/b + 1/c = 1 => 1/b + 1/c = 5/6
    #     2/b > 5/6 => b < 12/5 = 2.4. But b > 6. Contradiction.
    #   In general, for 3 terms with a < b < c: 3/a > 1 => a < 3. But a >= 5. Contradiction.
    #   So no 3-term solution with all denominators >= 5.
    #
    # - 4 terms: 1/a + 1/b + 1/c + 1/d = 1, a < b < c < d, all >= 5.
    #   4/a > 1 => a < 4. But a >= 5. Contradiction.
    #
    # - 5 terms: 5/a > 1 => a < 5. But a >= 5. Contradiction unless a=5 exactly gives 5/5=1, 
    #   but then remaining terms must sum to 0, impossible.
    #
    # Wait, let me reconsider. For n terms with a < b < ... < z, all >= 5:
    #   n/a > 1 => a < n. Since a >= 5, we need n > 5.
    #   So minimum n is 6? Let's check n=6: a < 6, so a=5.
    #   1/5 + 1/b + 1/c + 1/d + 1/e + 1/f = 1, 5 < b < c < d < e < f.
    #   1/b + 1/c + 1/d + 1/e + 1/f = 4/5.
    #   5/b > 4/5 => b < 25/4 = 6.25, so b=6.
    #   1/6 + 1/c + 1/d + 1/e + 1/f = 4/5 - 1/6 = 24/30 - 5/30 = 19/30.
    #   4/c > 19/30 => c < 120/19 ≈ 6.31, so c=7? But c > b=6, so c>=7.
    #   1/7 + 1/d + 1/e + 1/f = 19/30 - 1/7 = 133/210 - 30/210 = 103/210.
    #   3/d > 103/210 => d < 630/103 ≈ 6.11. But d > c=7. Contradiction.
    #
    # Let me try a different approach. Use known results or search more carefully.
    #
    # Actually, let's use a known representation. One well-known 5-term Egyptian fraction for 1:
    # 1/2 + 1/3 + 1/7 + 1/42 + 1/43 = 1? Let me check:
    # 1/2 + 1/3 = 5/6. 1/7 + 1/42 = 7/42 + 1/42 = 8/42 = 4/21. 5/6 + 4/21 = 35/42 + 8/42 = 43/42. Plus 1/43 is way over.
    #
    # Let me just use a computational approach within the function, but with a smarter search.
    #
    # Alternative: use the fact that we can find representations by solving the equation
    # using rational arithmetic and backtracking with strong pruning.
    #
    # Given the 60-second time limit, let's use a precomputed answer if we can find one,
    # or do a fast search.
    
    # Let's try to find a small representation by searching.
    # We'll use backtracking with the constraint that denominators are in increasing order
    # and all >= 5.
    
    from fractions import Fraction
    
    best = None
    best_len = 257
    
    def backtrack(remaining, min_denom, current):
        nonlocal best, best_len
        if len(current) >= best_len:
            return
        if remaining == 0:
            if len(current) < best_len:
                best_len = len(current)
                best = list(current)
            return
        if remaining < 0:
            return
        # Upper bound on how many more terms we can have
        # Each term is at most 1/min_denom, so we need at least ceil(remaining * min_denom) more terms
        # But also, if we have k more terms, the max sum is k/min_denom
        # So we need k/min_denom >= remaining => k >= remaining * min_denom
        import math
        min_needed = math.ceil(remaining * min_denom)
        if len(current) + min_needed >= best_len:
            return
        
        # Try denominators starting from min_denom
        # Upper bound on denominator: 1/d <= remaining => d >= 1/remaining
        # But also, we need enough room for remaining terms
        # Let's try denominators from min_denom up to some limit
        # The maximum denominator we should try: if this is the last term, d = 1/remaining
        # If not the last term, d can be larger, but we need at least 1 more term after, so:
        # 1/d + (something positive) = remaining => 1/d < remaining => d > 1/remaining
        # Actually for the current term, 1/d <= remaining, so d >= ceil(1/remaining)
        # But we also have d >= min_denom
        
        # Let's limit the search depth and breadth
        if len(current) > 12:
            return
            
        max_denom = min(10000000, max(min_denom, int(1 / remaining) + 1000) if remaining > 0 else 10000000)
        
        for d in range(min_denom, max_denom + 1):
            term = Fraction(1, d)
            if term > remaining:
                continue
            new_remaining = remaining - term
            if new_remaining == 0:
                current.append(d)
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                current.pop()
                return  # Found a solution with this d as last term, no need to try larger d
            elif new_remaining > 0:
                current.append(d)
                backtrack(new_remaining, d + 1, current)
                current.pop()
    
    backtrack(Fraction(1, 1), 5, [])
    
    if best is not None:
        return best
    else:
        # Fallback: use a known representation
        # Let's return a valid representation even if not minimal
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ... this is getting complex
        # Use the DP approach from the original as fallback
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
                return reached[target]
        raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
