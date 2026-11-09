import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Base solution: upper tail from 171 to 256
    # For any a<b<=c in [171,256]:
    # b+c >= 172+172 = 344 > 2*171 = 342, so b+c > 2a
    # b+c <= 256+256 = 512, and 3*171 = 513 > 512, so b+c < 3a
    # Thus a cannot divide b+c (no multiple of a falls in (2a, 3a) for a>=171)
    s = set(range(171, 257))
    best_set = set(s)
    best_size = len(s)

    def can_add(x, current_set):
        """Check if x can be added to current_set without violating the property.
        For every a<b<=c in (current_set ∪ {x}), a must NOT divide b+c.
        New violations can only involve x as one of a, b, or c."""
        elems = sorted(current_set)
        
        # Case 1: x is 'a' (smallest in triple)
        # Need: x does NOT divide b+c for all b,c in current_set with x<b<=c
        for b in elems:
            if b <= x:
                continue
            for c in elems:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle in triple)
        # Need: a does NOT divide x+c for all a in current_set with a<x, c in current_set with x<=c
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest in triple)
        # Need: a does NOT divide b+x for all a,b in current_set with a<b<=x
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def full_check(S):
        """Verify the entire set satisfies the property."""
        elems = sorted(S)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Greedily try to add elements from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline - 5:
            break
        if can_add(x, s):
            s.add(x)
    
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # Try multiple randomized restarts with different orderings
    for trial in range(50):
        if time.time() > deadline - 2:
            break
        
        # Start fresh from base
        s_trial = set(range(171, 257))
        
        # Try adding in a shuffled order, biased towards smaller numbers
        candidates = list(range(170, 0, -1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add(x, s_trial):
                s_trial.add(x)
        
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Local search: try to improve by swapping elements
    for _ in range(20):
        if time.time() > deadline - 1:
            break
        
        s_trial = set(best_set)
        improved = False
        
        # Try removing one element and adding a different one
        for rem in list(s_trial):
            if time.time() > deadline - 1:
                break
            s_trial.remove(rem)
            for add in range(1, 257):
                if add in s_trial or add == rem:
                    continue
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                    break
            s_trial.add(rem)
            if improved:
                break

    # Final validation
    if full_check(best_set):
        return list(best_set)
    
    # Fallback to known valid solution
    return list(range(171, 257))
