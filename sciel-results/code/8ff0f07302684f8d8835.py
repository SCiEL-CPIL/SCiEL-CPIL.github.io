import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    limit = 85  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Build conflict graph: for each number, which other numbers conflict with it
    # Two numbers conflict if they form a 3-AP with some third number in [1,N]
    # Actually, we need: three numbers a,b,c form a 3-AP if 2b = a+c
    # So for a pair (a,c) with a < c and (a+c) even, b = (a+c)/2 must not be in the set
    # This means: if we include both a and c, we cannot include b
    # Conflict: a and c conflict if b = (a+c)/2 is an integer in [1,N]
    
    # Precompute conflicts: for each x, list of y such that x and y cannot both be in a valid set
    # x and y conflict if there exists z such that {x,y,z} is a 3-AP
    # Case 1: x < z < y, so 2z = x+y, z = (x+y)/2 must be integer
    # Case 2: z < x < y, so 2x = z+y, z = 2x-y must be in [1,N]
    # Case 3: x < y < z, so 2y = x+z, z = 2y-x must be in [1,N]
    
    # Actually, the constraint is: no three distinct numbers a,b,c with 2b = a+c
    # So for a set S, for any a,c in S with a < c and (a+c) even, b = (a+c)/2 must not be in S
    # This means: a and c are "conflicting" in the sense that if both are in S, then b cannot be in S
    # But this is a 3-way constraint, not a 2-way one!
    
    # For branch-and-bound, we need to model this properly
    # Let's use a different approach: backtracking search with pruning
    
    # Precompute for each pair (a,c) with a < c and (a+c) even, the middle element b = (a+c)/2
    # If a and c are both selected, b is forbidden
    
    # Better approach: use a greedy + local search hybrid with better initialization
    
    def can_add(x, S):
        """Check if x can be added to S without forming a 3-AP."""
        # x is largest: a, b, x => b = (a+x)/2
        for a in S:
            if a >= x:
                continue
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # x is smallest: x, b, c => c = 2b-x, so b = (x+c)/2
        for c in S:
            if c <= x:
                continue
            s = x + c
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # x is middle: a, x, c => c = 2x-a
        for a in S:
            if a >= x:
                continue
            c = 2 * x - a
            if c > 0 and c in S:
                return False
        return True

    def get_sum(S):
        return sum(Fraction(1, x) for x in S)

    def greedy_fill(forbidden, S_start, order=None):
        """Greedy fill with optional custom ordering."""
        S = set(S_start)
        if order is None:
            order = all_nums
        for x in order:
            if x in forbidden or x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return S

    # Strategy 1: Multiple greedy initializations with different orderings
    best_set = None
    best_sum = Fraction(0)
    
    # Greedy 1: Natural order (1 to N) - favors small numbers
    S1 = greedy_fill(set(), set())
    sum1 = get_sum(S1)
    if best_set is None or sum1 > best_sum:
        best_set = S1
        best_sum = sum1
    
    # Greedy 2: Reverse order (N to 1) - favors large numbers (worse for reciprocal sum)
    S2 = greedy_fill(set(), set(), order=list(reversed(all_nums)))
    sum2 = get_sum(S2)
    if sum2 > best_sum:
        best_set = S2
        best_sum = sum2
    
    # Greedy 3: Random orderings
    for _ in range(10):
        if (time.time() - start_time) > limit:
            break
        random_order = all_nums[:]
        random.shuffle(random_order)
        S_rand = greedy_fill(set(), set(), order=random_order)
        sum_rand = get_sum(S_rand)
        if sum_rand > best_sum:
            best_set = S_rand
            best_sum = sum_rand
    
    # Strategy 2: Local search with swaps
    improved = True
    while improved and (time.time() - start_time) < limit:
        improved = False
        
        # Try removing large elements and adding small ones
        for x in sorted(best_set, reverse=True):
            if (time.time() - start_time) > limit:
                break
            
            # Try replacing x with smaller y
            for y in range(1, x):
                if y in best_set:
                    continue
                
                S_candidate = best_set - {x}
                if can_add(y, S_candidate):
                    S_candidate.add(y)
                    sum_new = get_sum(S_candidate)
                    if sum_new > best_sum:
                        best_set = S_candidate
                        best_sum = sum_new
                        improved = True
                        break
            if improved:
                break
        
        # Try removing one element and re-greedy filling
        for x in sorted(best_set, reverse=True):
            if (time.time() - start_time) > limit:
                break
            
            S_new = greedy_fill({x}, best_set - {x})
            sum_new = get_sum(S_new)
            
            if sum_new > best_sum:
                best_set = S_new
                best_sum = sum_new
                improved = True
                break
    
    return list(best_set)
