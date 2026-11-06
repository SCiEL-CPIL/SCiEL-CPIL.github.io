import time

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    n = 256
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S = sorted(S)
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to set S while maintaining validity."""
        if x in S:
            return True
        S_sorted = sorted(S)
        # x as 'a': for all b,c in S with x<b<=c, check x ∤ (b+c)
        for i in range(len(S_sorted)):
            b = S_sorted[i]
            if b <= x:
                continue
            for j in range(i, len(S_sorted)):
                c = S_sorted[j]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        # x as 'b': for all a,c in S with a<x<=c, check a ∤ (x+c)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            if a >= x:
                break
            for j in range(len(S_sorted)):
                c = S_sorted[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        # x as 'c': for all a,b in S with a<b<=x, check a ∤ (b+x)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            for j in range(i+1, len(S_sorted)):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        return True

    def greedy_extend(S, candidates):
        """Try to add elements from candidates to S greedily."""
        S = set(S)
        for x in candidates:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    # Start with a known valid set: Upper tail [171, 256]
    # This set is valid because for any a >= 171, b,c >= 172, b+c >= 344.
    # Multiples of a in [344, 511]:
    # If a=171, mult=342 (too small), 513 (too big).
    # If a=172, mult=344 (too small, min sum is 173+173=346? No, b,c >= a+1? No, b,c in S, so b,c >= 172.
    # Wait, if a=172, b,c can be 172? No, a < b. So b >= 173.
    # Min sum 173+173=346. 344 is too small. Next 516 too big.
    # So yes, [171, 256] is valid.
    
    S = set(range(171, 257))
    
    # Try to extend by adding smaller elements
    candidates = list(range(1, 171))
    
    # Try ascending
    S1 = greedy_extend(S, candidates)
    
    # Try descending
    S2 = greedy_extend(S, sorted(candidates, reverse=True))
    
    # Try random shuffles if time permits? No, keep it simple.
    
    best_set = S1 if len(S1) > len(S2) else S2
    
    # Final check
    if not is_valid_set(best_set):
        best_set = set(range(171, 257))
        
    return sorted(best_set)
