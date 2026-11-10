import time
from fractions import Fraction

def construct():
    start_time = time.time()
    limit = 85  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    def can_add(x, S):
        # Check if x can be added to S without forming a 3-AP
        # 1. x is largest: a, b, x => 2b = a+x => b = (a+x)/2
        #    a in S, a < x. b must be integer and in S.
        for a in S:
            if a >= x:
                continue
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # 2. x is smallest: x, b, c => 2b = x+c => c = 2b-x
        #    c in S, c > x. b must be integer.
        #    Alternatively: x is smallest, so b = (x+c)/2.
        #    Let's iterate c in S where c > x.
        for c in S:
            if c <= x:
                continue
            s = x + c
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # 3. x is middle: a, x, c => 2x = a+c => c = 2x-a
        #    a in S, a < x. c must be in S.
        for a in S:
            if a >= x:
                continue
            c = 2 * x - a
            if c > 0 and c in S:
                return False
        return True

    def get_sum(S):
        return sum(Fraction(1, x) for x in S)

    def greedy_fill(forbidden, S_start):
        # S is a set
        # forbidden is a set of numbers we cannot add
        S = set(S_start)
        for x in all_nums:
            if x in forbidden:
                continue
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return S

    # 1. Initial Greedy
    # Start with empty set, no forbidden
    S_best = greedy_fill(set(), set())
    sum_best = get_sum(S_best)
    
    # 2. Enhanced Local Search
    # We try:
    # a) Removing an element and re-greedy filling.
    # b) Swapping an element x in S with an element y not in S (y < x preferred).
    
    improved = True
    while improved and (time.time() - start_time) < limit:
        improved = False
        
        # Strategy A: Swap
        # Iterate through elements in S. To maximize gain, we want to replace large x with small y.
        # So iterate x in descending order (largest first).
        for x in sorted(S_best, reverse=True):
            if (time.time() - start_time) > limit:
                return list(S_best)
            
            # Try replacing x with y
            # y must be in [1, N] and not in S_best (except x which is being removed)
            # To ensure gain, we look for y < x. If no y < x works, we might still try y > x if it allows adding other small numbers later?
            # For now, let's focus on y < x for direct gain.
            for y in range(1, x):
                if y in S_best:
                    continue
                
                S_candidate = S_best - {x}
                if can_add(y, S_candidate):
                    S_candidate.add(y)
                    sum_new = get_sum(S_candidate)
                    if sum_new > sum_best:
                        S_best = S_candidate
                        sum_best = sum_new
                        improved = True
                        break # Restart inner loop? Or just restart outer?
                        # Breaking inner loop and restarting outer is safer to pick up new interactions.
            if improved:
                break
        
        if improved:
            continue

        # Strategy B: Remove and Re-greedy (Current method, kept as backup)
        # This can find solutions where removing one element allows adding MULTIPLE small elements.
        for x in sorted(S_best, reverse=True):
            if (time.time() - start_time) > limit:
                return list(S_best)
            
            forbidden = {x}
            S_new = greedy_fill(forbidden, S_best - {x})
            
            sum_new = get_sum(S_new)
            
            if sum_new > sum_best:
                S_best = S_new
                sum_best = sum_new
                improved = True
                break
        
    return list(S_best)
