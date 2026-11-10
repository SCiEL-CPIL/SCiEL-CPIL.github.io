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
    
    # 2. Local Search
    # We try removing each element x from S_best, then re-greedy-fill.
    # If the new sum is higher, we accept it.
    # We repeat this until no improvement is found or time runs out.
    
    improved = True
    while improved and (time.time() - start_time) < limit:
        improved = False
        # Sort elements by value descending (largest x first) to prioritize removing "expensive" blockers
        # Actually, removing any x might allow adding many small y's.
        # Let's just iterate through all x in S_best.
        # To ensure we don't miss improvements, we should restart the scan if we find one.
        
        for x in list(S_best):
            if (time.time() - start_time) > limit:
                return list(S_best)
            
            # Try removing x
            forbidden = {x}
            S_new = greedy_fill(forbidden, S_best - {x})
            
            sum_new = get_sum(S_new)
            
            if sum_new > sum_best:
                S_best = S_new
                sum_best = sum_new
                improved = True
                # Restart the for loop to re-evaluate with the new S_best
                # This is a simple hill climbing step.
                break
        
    return list(S_best)
