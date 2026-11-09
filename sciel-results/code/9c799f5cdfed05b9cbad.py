import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # 1. Greedy Construction
    selected = []
    for x in range(1, 257):
        # Check if x forms 3-AP with any pair in selected
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a = selected[i]
                b = selected[j]
                # x is the new element (largest so far? No, x > a, b)
                # So x can only be the end of the AP: a, b, x -> 2b = a + x? No.
                # AP: a, b, c.
                # If we add x, and a, b are in S.
                # Since x > a, b, x is the largest.
                # So the AP must be a, b, x with 2b = a + x? No.
                # The AP is {a, b, x}.
                # Midpoint is b? Then a + x = 2b.
                # Midpoint is a? Then b + x = 2a (Impossible as x>a).
                # Midpoint is x? Then a + b = 2x (Impossible as x>a,b).
                # So only check: a + x == 2*b OR b + x == 2*a?
                # Wait, if a < b < x.
                # APs:
                # 1. a, b, x: 2b = a + x
                # 2. a, x, b: Impossible (x > b)
                # 3. b, a, x: Impossible (b > a)
                # 4. x, a, b: Impossible
                # 5. x, b, a: Impossible
                # 6. b, x, a: Impossible
                # 
                # Wait, is it possible that a, x, b is an AP? No, order matters for the set, but AP is a property of the values.
                # The set {a, b, x} is a 3-AP if one is the midpoint of the other two.
                # Since x is the largest, x cannot be the midpoint.
                # So either a is midpoint: 2a = b + x (Impossible x > a, b > a -> b+x > 2a)
                # Or b is midpoint: 2b = a + x.
                # 
                # So we only need to check: 2*b == a + x for a < b in S.
                if 2 * b == a + x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    S = set(selected)
    
    # 2. Local Search
    # We want to maximize sum(1/x).
    # Swap x in S with y not in S if 1/y > 1/x (i.e., y < x) and S' is valid.
    
    def is_valid_add(S, y):
        # Check if S U {y} is 3-AP free.
        # y is new. Check if y forms AP with any pair in S.
        # Pairs (a, b) in S.
        # y, a, b form AP.
        # y can be mid, start, or end.
        for a in S:
            for b in S:
                if a == b:
                    continue
                # 1. y is mid: 2y = a + b
                if 2 * y == a + b:
                    return False
                # 2. y is end: 2a = y + b  => y = 2a - b
                # 3. y is start: 2b = y + a => y = 2b - a
                # Combined: a, b, y is AP if 2*mid = start + end.
                # Just check the 3 conditions.
                if 2 * a == y + b:
                    return False
                if 2 * b == y + a:
                    return False
        return True

    improved = True
    while improved:
        improved = False
        # Sort S descending to try replacing large elements first
        for x in sorted(S, reverse=True):
            if time.time() > deadline:
                return sorted(S)
            
            S_temp = S - {x}
            best_y = None
            
            # Find smallest y < x such that S_temp U {y} is valid
            # We can iterate y from 1 to x-1
            for y in range(1, x):
                if y in S_temp:
                    continue
                
                if is_valid_add(S_temp, y):
                    best_y = y
                    break # Smallest y is best
            
            if best_y is not None:
                S.remove(x)
                S.add(best_y)
                improved = True
                # Restart loop to account for changes
                break
    
    return sorted(S)
