import time
import random

def is_3ap_free(S):
    S_set = set(S)
    for a in S:
        for b in S:
            if a != b:
                c = 2*b - a
                if c in S_set and c != a and c != b:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 55.0 # 55 seconds to be safe
    
    # Greedy initialization
    S = set()
    for x in range(1, 257):
        if time.time() - start_time > deadline:
            break
        # Check if x can be added
        can_add = True
        for a in S:
            # x is middle: a, x, c => c = 2x - a
            c = 2*x - a
            if c in S and c != x:
                can_add = False
                break
            # x is end: a, b, x => b = (a+x)/2, a = 2b - x
            # Check if there exists b in S such that a = 2b - x
            # This is equivalent to checking if (a+x) is even and (a+x)/2 in S
            # Or simply: for each b in S, check if 2b - x in S
            # Let's just do the O(|S|) check for both cases
        # The above loop is incomplete. Let's use the efficient check function.
        
    def can_add(x, S_set):
        for a in S_set:
            # Case 1: a, x, c => c = 2x - a
            c = 2*x - a
            if c in S_set and c != x:
                return False
            # Case 2: x, b, c => c = 2b - x. Here b is in S.
            # We need to check if there is any b in S such that 2b - x is in S.
            # This is not covered by the above loop directly.
            # Let's rephrase:
            # A 3-AP involving x and two elements in S must have x as one of the terms.
            # If x is the middle term: a, x, c. Then a, c in S and a+c=2x.
            # If x is the first term: x, b, c. Then b, c in S and c-b = b-x => c = 2b - x.
            # If x is the last term: a, b, x. Then a, b in S and b-a = x-b => a = 2b - x.
            
            # Check if x is middle:
            # Already checked c = 2x - a. If c in S, conflict.
            
            # Check if x is first or last:
            # This requires checking pairs in S.
            # To keep it O(|S|), we can do:
            # For each b in S, check if 2b - x in S.
            # But we are already iterating over a in S.
            # Let's just do two loops for clarity and speed.
        return True

    # Let's rewrite the check to be efficient and correct.
    def can_add(x, S_set):
        # Check if x forms a 3-AP with any two elements in S_set
        # 1. x is the middle term: a, x, c. a, c in S_set. a+c=2x.
        for a in S_set:
            c = 2*x - a
            if c in S_set and c != x:
                return False
        # 2. x is the end term: a, b, x or x, b, c.
        # This means there exist b, c in S_set such that c - b = b - x (x is first)
        # or b - a = x - b (x is last).
        # In both cases, x is one of the ends.
        # If x is first: x, b, c => c = 2b - x. b, c in S_set.
        # If x is last: a, b, x => a = 2b - x. a, b in S_set.
        # So we just need to check if for any b in S_set, 2b - x is in S_set.
        for b in S_set:
            c = 2*b - x
            if c in S_set and c != x and c != b:
                return False
        return True

    # Greedy initialization
    S = set()
    for x in range(1, 257):
        if time.time() - start_time > deadline:
            break
        if can_add(x, S):
            S.add(x)
    
    S_set = S
    
    # Local search
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # Try to add any element
        for x in range(1, 257):
            if time.time() - start_time > deadline:
                break
            if x not in S_set and can_add(x, S_set):
                S_set.add(x)
                improved = True
                # Restart the loop to try more additions
                break
        
        # Try to swap
        for x in list(S_set):
            if time.time() - start_time > deadline:
                break
            S_set.remove(x)
            for y in range(1, 257):
                if time.time() - start_time > deadline:
                    break
                if y not in S_set and can_add(y, S_set):
                    # Check if swap improves the score
                    if 1.0/y > 1.0/x:
                        S_set.add(y)
                        improved = True
                        break
            if improved:
                break
            # If no swap found, put x back
            if x not in S_set:
                S_set.add(x)
        
        # Try to remove one and add two
        # This is more complex. Let's skip for now and see if simple swaps are enough.
        
    return list(S_set)
