"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Known minimal Egyptian fraction for 1 with denominators > 4:
    # 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5  -- wait, that's not 1.
    # Let me recompute:
    # 1/5 + 1/6 + 1/30 = (6+5+1)/30 = 12/30 = 2/5. Not 1.
    # 
    # Let me find a correct 3-term representation:
    # 1/5 + 1/5 + 1/5 = 3/5 (not allowed, distinct)
    # 
    # 1/6 + 1/3 + ... no, 3 < 5.
    #
    # Let me try: 1/a + 1/b + 1/c = 1 with 5 <= a < b < c
    # 
    # 1/5 + 1/6 + 1/x = 1 => 1/x = 1 - 11/30 = 19/30 => x = 30/19 (not integer)
    # 1/5 + 1/7 + 1/x = 1 => 1/x = 1 - 12/35 = 23/35 => x = 35/23 (not integer)
    # 1/5 + 1/8 + 1/x = 1 => 1/x = 1 - 13/40 = 27/40 => x = 40/27 (not integer)
    # 1/5 + 1/9 + 1/x = 1 => 1/x = 1 - 14/45 = 31/45 => x = 45/31 (not integer)
    # 1/5 + 1/10 + 1/x = 1 => 1/x = 1 - 1/2 = 1/2 => x = 2 (too small)
    #
    # 1/6 + 1/7 + 1/x = 1 => 1/x = 1 - 13/42 = 29/42 => x = 42/29 (not integer)
    # 1/6 + 1/8 + 1/x = 1 => 1/x = 1 - 7/24 = 17/24 => x = 24/17 (not integer)
    # 1/6 + 1/9 + 1/x = 1 => 1/x = 1 - 5/18 = 13/18 => x = 18/13 (not integer)
    # 1/6 + 1/10 + 1/x = 1 => 1/x = 1 - 8/30 = 22/30 = 11/15 => x = 15/11 (not integer)
    #
    # 1/7 + 1/8 + 1/x = 1 => 1/x = 1 - 15/56 = 41/56 => x = 56/41 (not integer)
    # 1/7 + 1/9 + 1/x = 1 => 1/x = 1 - 16/63 = 47/63 => x = 63/47 (not integer)
    #
    # Let me try 4 terms:
    # 1/5 + 1/6 + 1/7 + 1/x = 1
    # 1/5 + 1/6 + 1/7 = 42/210 + 35/210 + 30/210 = 107/210
    # 1/x = 103/210 => x = 210/103 (not integer)
    #
    # 1/5 + 1/6 + 1/8 + 1/x = 1
    # 1/5 + 1/6 + 1/8 = 24/120 + 20/120 + 15/120 = 59/120
    # 1/x = 61/120 => x = 120/61 (not integer)
    #
    # 1/5 + 1/6 + 1/9 + 1/x = 1
    # 1/5 + 1/6 + 1/9 = 18/90 + 15/90 + 10/90 = 43/90
    # 1/x = 47/90 => x = 90/47 (not integer)
    #
    # 1/5 + 1/6 + 1/10 + 1/x = 1
    # 1/5 + 1/6 + 1/10 = 6/30 + 5/30 + 3/30 = 14/30 = 7/15
    # 1/x = 8/15 => x = 15/8 (not integer)
    #
    # 1/5 + 1/7 + 1/8 + 1/x = 1
    # 1/5 + 1/7 + 1/8 = 56/280 + 40/280 + 35/280 = 131/280
    # 1/x = 149/280 => x = 280/149 (not integer)
    #
    # Let me try a different approach. The classic result is that 1 cannot be written
    # as a sum of fewer than 3 unit fractions with all denominators > 4, but let me
    # verify this is actually true and find a valid representation.
    #
    # Actually, let me reconsider. The task says denominators in [5, 10000000].
    # Let me check if 3 terms is possible at all.
    #
    # For 1/a + 1/b + 1/c = 1 with 5 <= a < b < c:
    # We need 1/a >= 1/3, so a <= 3. But a >= 5. Contradiction!
    # So 3 terms is IMPOSSIBLE with all denominators >= 5.
    #
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 1 with 5 <= a < b < c < d
    # We need 1/a >= 1/4, so a <= 4. But a >= 5. Contradiction!
    # So 4 terms is also IMPOSSIBLE.
    #
    # For 5 terms: 1/a + 1/b + 1/c + 1/d + 1/e = 1 with 5 <= a < b < c < d < e
    # We need 1/a >= 1/5, so a <= 5. Thus a = 5.
    # Then 1/b + 1/c + 1/d + 1/e = 4/5
    # We need 1/b >= 1/4, so b <= 4. But b > a = 5. Contradiction? No wait...
    # Actually we need the sum of 4 unit fractions to be 4/5.
    # 1/b + 1/c + 1/d + 1/e = 4/5
    # The largest possible value is 1/6 + 1/7 + 1/8 + 1/9 = 94/252 + ... let me compute
    # 1/6 + 1/7 + 1/8 + 1/9 = (840 + 720 + 630 + 560)/5040 = 2750/5040 = 275/504 ≈ 0.546
    # 4/5 = 0.8. So we need larger fractions, meaning smaller denominators.
    # But b > 5, so b >= 6.
    # 1/6 + 1/7 + 1/8 + 1/9 ≈ 0.546 < 0.8.
    # 1/6 + 1/6 + 1/6 + 1/6 = 2/3 ≈ 0.667 < 0.8 (but not distinct)
    #
    # So 5 terms might also be impossible. Let me check more carefully.
    # Max sum with 5 distinct denominators >= 5: 1/5 + 1/6 + 1/7 + 1/8 + 1/9
    # = (504 + 420 + 360 + 315 + 280)/2520 = 1879/2520 ≈ 0.746 < 1.
    # So 5 terms is IMPOSSIBLE.
    #
    # 6 terms: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10
    # = (504 + 420 + 360 + 315 + 280 + 252)/2520 = 2131/2520 ≈ 0.846 < 1. Still not enough.
    #
    # We need more terms or larger fractions. But denominators are fixed >= 5.
    # The maximum sum with n distinct denominators >= 5 is 1/5 + 1/6 + ... + 1/(n+4).
    # We need this sum >= 1.
    #
    # Harmonic series: H(n) - H(4) >= 1
    # H(n) - (1 + 1/2 + 1/3 + 1/4) >= 1
    # H(n) - 25/12 >= 1
    # H(n) >= 37/12 ≈ 3.0833
    # H(10) = 7381/2520 ≈ 2.929
    # H(11) = 83711/27720 ≈ 3.020
    # H(12) = 86021/27720 ≈ 3.103 >= 3.0833
    # So we need at least n = 12 terms where the smallest denominator is 5.
    # But wait, that's the maximum sum. We need exactly 1, not at least 1.
    #
    # Let me just use a known valid representation. The simplest approach:
    # Use the greedy Egyptian fraction algorithm to decompose 1 with denominators >= 5.
    #
    # Actually, let me just hardcode a known valid solution.
    # 1 = 1/5 + 1/6 + 1/10 + 1/15 + 1/30
    # Check: 1/5 + 1/6 + 1/10 + 1/15 + 1/30
    # = 6/30 + 5/30 + 3/30 + 2/30 + 1/30 = 17/30. Not 1.
    #
    # Let me use the greedy method properly:
    # 1/1: need denominator >= 5, so can't use 1.
    # Greedy: ceil(1/1) = 1, but 1 < 5. So greedy doesn't directly work.
    #
    # Let me try: 1 = 1/5 + 4/5
    # 4/5 = 1/2 + 3/10... 2 < 5.
    #
    # Let me just find any valid set. I'll use a small search.
    # Actually, let me verify: [5, 6, 10, 15, 30] sum = 17/30. Not 1.
    #
    # Try: [5, 6, 7, 14, 35, ...] this is getting complex.
    #
    # Let me just implement a quick search for a valid solution and hardcode it.
    # For now, return a known valid large representation.
    
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + ...
    # This won't sum to exactly 1 easily.
    
    # Let me use: 1/5 + 1/5 + 1/5 + 1/5 + 1/5 = 1, but not distinct.
    
    # OK let me just do a proper search in code.
    from fractions import Fraction
    from itertools import combinations
    import time
    
    start = time.time()
    deadline = 85.0  # Leave 5s margin
    
    # Try to find minimal number of distinct denominators >= 5 summing to 1
    # We know >= 6 terms are needed (5 terms max sum < 1)
    
    # Let's try 6 terms first with a search
    # 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 1
    # a < b < c < d < e < f, all >= 5
    
    # Max sum with 6 terms: 1/5+1/6+1/7+1/8+1/9+1/10 = 2131/2520 < 1
    # So 6 terms is impossible!
    
    # 7 terms: max sum = 1/5+...+1/11
    # = (504+420+360+315+280+252+240)/2520 = 2371/2520 ≈ 0.941 < 1
    # Still not enough!
    
    # 8 terms: max sum = 1/5+...+1/12
    # Additional: 1/12 = 210/2520
    # Sum = 2371/2520 + 210/2520 = 2581/2520 > 1
    # So 8 terms is possible in principle.
    
    # Let me search for 8-term solutions
    # 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h = 1
    
    # Use a backtracking search
    def search(min_d, remaining, terms_found, depth, max_depth):
        if time.time() > deadline:
            return None
        if depth == max_depth:
            if remaining == 0:
                return terms_found
            return None
        if remaining <= 0:
            return None
        # Max possible sum from remaining terms
        max_possible = sum(1/(min_d + i) for i in range(max_depth - depth))
        if max_possible < remaining:
            return None
        
        # Try denominators
        for d in range(min_d, min_d + 200):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > remaining:
                break
            new_remaining = remaining - frac
            if new_remaining > 0:
                result = search(d + 1, new_remaining, terms_found + [d], depth + 1, max_depth)
                if result is not None:
                    return result
            elif new_remaining == 0:
                return terms_found + [d]
        return None
    
    # Try 8 terms
    result = search(5, Fraction(1, 1), [], 0, 8)
    if result is not None:
        if time.time() < deadline:
            return result
        else:
            return result
    
    # If 8 terms didn't work quickly, try 9
    result = search(5, Fraction(1, 1), [], 0, 9)
    if result is not None:
        return result
    
    # Fallback: use a known larger representation
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + ...
    # Let me just return a valid (if not minimal) solution
    # Use the identity: 1 = 1/5 + 1/6 + 1/10 + 1/15 + 1/30 + ... no
    #
    # Actually let me just construct one:
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 19/30
    # 19/30 = 1/7 + 67/210
    # 67/210 = 1/8 + 129/840
    # 129/840 = 1/9 + 167/2520
    # 167/2520 = 1/10 + 1147/25200
    # 1147/25200 = 1/11 + 10377/277200
    # 10377/277200 = 1/12 + 96213/3326400
    # ... this is getting large but each step increases denominator
    #
    # Let me just verify [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    # Sum = H(40) - H(4) = H(40) - 25/12
    # This won't be exactly 1.
    
    # Let me just return the search result or a fallback
    # For safety, return a valid but possibly non-minimal solution
    # 1/5 + 1/6 + 1/10 + 1/15 + 1/30 + 1/6 + 1/10 + 1/15 + 1/30... no, must be distinct
    
    # Fallback: use [5, 6, 10, 15, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # This is getting silly. Let me just ensure we have a valid answer.
    
    # The search should have found something. If not, return a minimal valid set.
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (invalid, not distinct)
    
    # Let me just return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # and hope the sum is close enough? No, must be exactly 1.
    
    # I'll just return the search result. If search failed, raise.
    raise RuntimeError("search failed to find valid representation")

# EVOLVE-BLOCK-END
