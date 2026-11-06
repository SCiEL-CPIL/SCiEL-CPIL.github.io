"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 80  # 80s wall-clock guard (90s limit, 10s margin)
    
    # The task: find 1..256 distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1; minimize their count.
    # 
    # Key insight: we want the MINIMUM number of terms.
    # Lower bound: since each denominator >= 5, each term <= 1/5.
    # So we need at least 5 terms (5 * 1/5 = 1).
    # Can we do it with exactly 5 terms?
    # 1/a + 1/b + 1/c + 1/d + 1/e = 1 with 5 <= a < b < c < d < e
    # 
    # Let's try to find a 5-term solution first.
    # If 1/a + 1/b + 1/c + 1/d + 1/e = 1, with a < b < c < d < e
    # 
    # Known identity: 1/2 + 1/3 + 1/7 + 1/42 = 1 (4 terms, but 2 < 5)
    # 1/3 + 1/4 + 1/5 + 1/6 + 1/20 = 1? Let's check: 20+15+12+10+3 = 60/60 = 1. YES!
    # But 3 < 5, so that doesn't work directly.
    #
    # Let's search for 5-term solutions with all denominators >= 5.
    #
    # Strategy: try small values of a (smallest denominator) and work outward.
    # For a=5: 1/5 + 1/b + 1/c + 1/d + 1/e = 1 => 1/b + 1/c + 1/d + 1/e = 4/5
    #   With b > 5: 1/b + 1/c + 1/d + 1/e <= 4/6 = 2/3 < 4/5. So a=5 impossible with 5 terms.
    # 
    # Actually wait, b > a = 5, so b >= 6. Max of 1/b+1/c+1/d+1/e with b=6,c=7,d=8,e=9:
    #   1/6+1/7+1/8+1/9 = 0.1667+0.1429+0.125+0.1111 = 0.5456 < 0.8. So a=5 doesn't work.
    #
    # For a=6: 1/6 + rest = 1 => rest = 5/6. b>=7: max 1/7+1/8+1/9+1/10 = 0.4395 < 5/6. No.
    #
    # For a=7: rest = 6/7. b>=8: max 1/8+1/9+1/10+1/11 = 0.3977 < 6/7. No.
    #
    # So 5 terms is impossible? Let me reconsider.
    # Actually I need to be more careful. The denominators are DISTINCT and >= 5.
    #
    # Let's try 6 terms: 1/a+1/b+1/c+1/d+1/e+1/f = 1, 5<=a<b<c<d<e<f
    #
    # Try a=5: rest = 4/5. b>=6: 1/6+1/7+1/8+1/9+1/10 = 0.6111. Need 4/5=0.8. Not enough.
    # Wait, that's only 5 more terms but we need exactly 5 more (total 6). 
    # 1/6+1/7+1/8+1/9+1/10 = 0.6111 < 0.8. So a=5 with 6 terms doesn't work either?
    # Hmm, but we could use larger denominators... no wait, we need the SUM to be 4/5.
    # With b=6,c=7,d=8,e=9,f=10: sum = 0.6111 < 0.8. The maximum possible sum of 5 terms 
    # with denominators > 5 is achieved with the smallest denominators: 6,7,8,9,10.
    # So 5 terms after a=5 can't reach 4/5. a=5 is out for 6 terms too.
    #
    # Try a=6: rest = 5/6 ≈ 0.833. b>=7: max 1/7+1/8+1/9+1/10+1/11 = 0.4972 < 0.833. No.
    #
    # This suggests we need many more terms. Let me think differently.
    #
    # Actually, let me reconsider. The problem says "minimize their count" and the 
    # evaluator measures the raw value (lower is better). So we want the FEWEST terms.
    #
    # Let me use a systematic search. For small k, I'll try to find solutions.
    # 
    # Alternative approach: use the fact that we can always find a solution by
    # starting with 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (but these aren't distinct).
    # We need distinct denominators.
    #
    # Let me try a greedy/backtracking approach for small k.
    
    best_solution = None
    best_len = 257  # upper bound
    
    def time_left():
        return deadline - time.time()
    
    # Try to find solutions with increasing number of terms
    # Start with k=5,6,7,... and stop when we find a solution
    
    # For efficiency, use a DFS with pruning
    # State: current sum (as fraction), current list of denominators, next minimum denominator
    
    # Let's use a different strategy: enumerate known identities and small cases
    # 
    # Actually, let me think about this more carefully.
    # 
    # For k terms with denominators d1 < d2 < ... < dk, all >= 5:
    # 1/d1 + 1/d2 + ... + 1/dk = 1
    #
    # d1 >= 5, so 1/d1 <= 1/5, meaning we need at least 5 terms.
    # But as shown, 5 and 6 terms seem infeasible.
    #
    # Let me try k=7: d1=5, rest sum = 4/5
    #   d2=6: rest = 4/5 - 1/6 = 24/30 - 5/30 = 19/30
    #   d3=7: rest = 19/30 - 1/7 = 133/210 - 30/210 = 103/210
    #   d4=8: rest = 103/210 - 1/8 = 824/1680 - 210/1680 = 614/1680 = 307/840
    #   d5=9: rest = 307/840 - 1/9 = 2763/7560 - 980/7560 = 1783/7560
    #   d6=10: rest = 1783/7560 - 1/10 = 17830/75600 - 7560/75600 = 10270/75600 = 1027/7560
    #   d7: 1/d7 = 1027/7560 => d7 = 7560/1027 ≈ 7.36. Not integer.
    #
    # This is getting complex. Let me use a programmatic search.
    
    # DFS approach with fraction arithmetic
    def dfs(rem_num, rem_den, min_d, current, depth, max_depth):
        nonlocal best_solution, best_len
        if time_left() < 0.5:
            return
        
        if depth == max_depth:
            if rem_num == rem_den and current:
                if len(current) < best_len:
                    best_len = len(current)
                    best_solution = list(current)
            return
        
        if rem_num <= 0:
            return
        
        # Normalize
        g = gcd(rem_num, rem_den)
        rn, rd = rem_num // g, rem_den // g
        
        if rn == 0:
            if current and len(current) < best_len:
                best_len = len(current)
                best_solution = list(current)
            return
        
        # Upper bound on next denominator: 1/d >= 1/max_possible
        # We need sum of remaining (max_depth - depth) terms to equal rn/rd
        # Each term <= 1/min_d, so max sum = (max_depth - depth) / min_d
        # We need rn/rd <= (max_depth - depth) / min_d
        # Also, 1/d >= rn/rd / (max_depth - depth) => d <= rd * (max_depth - depth) / rn
        
        remaining_terms = max_depth - depth
        if remaining_terms <= 0:
            return
        
        # Max denominator: 1/d >= rn/rd / remaining_terms => d <= rd * remaining_terms / rn
        max_d = rd * remaining_terms // rn + 1
        # Also d <= 10000000
        max_d = min(max_d, 10000000)
        
        # Min denominator: d >= min_d
        # Also 1/d <= rn/rd => d >= rd/rn
        min_d_actual = max(min_d, rd // rn + 1 if rd % rn != 0 else rd // rn)
        # Actually 1/d <= rn/rd means d >= rd/rn. Since d is integer, d >= ceil(rd/rn)
        min_d_actual = max(min_d, (rd + rn - 1) // rn)
        
        if min_d_actual > max_d:
            return
        
        # Additional pruning: with remaining terms, the minimum possible sum
        # is achieved with the largest denominators. The maximum possible sum
        # is achieved with the smallest denominators (starting from min_d_actual).
        # Max sum = sum of 1/i for i = min_d_actual to min_d_actual + remaining_terms - 1
        # This is hard to compute exactly, so use a looser bound:
        # max sum <= remaining_terms / min_d_actual
        # We need rn/rd <= remaining_terms / min_d_actual
        # i.e., rn * min_d_actual <= rd * remaining_terms
        if rn * min_d_actual > rd * remaining_terms:
            return
        
        # Min sum = sum of 1/i for i = max_d - remaining_terms + 1 to max_d
        # Approximate: min sum >= remaining_terms / max_d
        # We need rn/rd >= remaining_terms / max_d
        # i.e., rn * max_d >= rd * remaining_terms
        if rn * max_d < rd * remaining_terms:
            return
        
        for d in range(min_d_actual, max_d + 1):
            if time_left() < 0.1:
                return
            # 1/d + rest = rn/rd => rest = rn/rd - 1/d = (rn*d - rd) / (rd*d)
            new_num = rn * d - rd
            if new_num <= 0:
                continue
            new_den = rd * d
            current.append(d)
            dfs(new_num, new_den, d + 1, current, depth + 1, max_depth)
            current.pop()
            
            # Early stop if we found a good solution
            if best_solution is not None and best_len <= 5:
                return
    
    # Try increasing k from 5 upward
    # But first, let's try some known constructions
    
    # Known: 1/3 + 1/4 + 1/5 + 1/6 + 1/20 = 1 (but 3 < 5)
    # Let's try to find solutions programmatically
    
    # Start DFS for k=5
    for k in range(5, 30):
        if time_left() < 1.0:
            break
        dfs(1, 1, 5, [], 0, k)
        if best_solution is not None:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use a known large solution
    # 1/5 + 1/6 + 1/7 + ... can be adjusted
    # 
    # Use the identity: 1/n = 1/(n+1) + 1/(n(n+1))
    # Starting from 1/5:
    # 1/5 = 1/6 + 1/30
    # 1/6 = 1/7 + 1/42
    # etc.
    
    # Simple fallback: find ANY valid solution
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (not distinct)
    # 
    # Use: 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + ... + adjust
    #
    # Let's just do a simple search for any solution
    # Start with denominators 5, 6, 7, ... and add terms until sum >= 1, then adjust
    
    # Greedy: pick smallest available denominator that doesn't overshoot
    result = []
    rem_num, rem_den = 1, 1
    next_d = 5
    while rem_num > 0 and time_left() > 1.0:
        # Find smallest d >= next_d such that 1/d <= rem_num/rem_den
        # i.e., d >= rem_den / rem_num
        min_needed = rem_den // rem_num + (1 if rem_den % rem_num != 0 else 0)
        d = max(next_d, min_needed)
        if d > 10000000:
            break
        # Check if 1/d <= rem
        if rem_num * d <= rem_den:
            result.append(d)
            # new rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            rem_num = rem_num * d - rem_den
            rem_den = rem_den * d
            next_d = d + 1
            if rem_num == 0:
                break
        else:
            next_d += 1
    
    if rem_num == 0 and len(result) >= 1 and len(result) <= 256:
        if all(d >= 5 and d <= 10000000 for d in result):
            if len(set(result)) == len(result):
                return result
    
    # Last resort: return a known valid solution
    # Let's construct one: use the splitting identity repeatedly
    # 1 = 1/5 + 4/5
    # 4/5 = 4/5 ... need to split into distinct unit fractions
    # 
    # Actually, let me just return a hardcoded valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # Sum ≈ 0.2+0.1667+0.1429+0.125+0.1111+0.1+0.0909+0.0833+0.0769+0.0714+0.0667+0.0625+0.0588+0.0556+0.0526+0.05
    # ≈ 1.4542. Too much.
    
    # Let me try fewer terms with larger denominators
    # We need sum = 1. With k terms, average term = 1/k, so average denominator = k.
    # For k=10: avg denom = 10, but min is 5. 
    # For k=20: avg denom = 20.
    
    # Just return a valid solution found by the greedy method or a simple one
    # Let's verify: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24
    # This will exceed 1.
    
    # Use the greedy result if valid, otherwise raise
    if result and rem_num == 0:
        return result
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END

