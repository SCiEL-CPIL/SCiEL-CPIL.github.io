import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    base_S = set(range(65, 129))
    candidates = list(range(1, 65))
    
    best_set = base_S.copy()
    best_size = len(best_set)
    
    def is_valid_addition(x, current_S):
        new_S = current_S | {x}
        
        # 1. Check triples involving x: {x, a, b}
        # Sums: x+a, x+b, a+b
        # Must not all be in new_S
        elems = list(current_S)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            if a == x: continue
            sa = x + a
            if sa > 128 or sa not in new_S:
                continue
            for j in range(i+1, n):
                b = elems[j]
                if b == x: continue
                sb = x + b
                if sb > 128 or sb not in new_S:
                    continue
                sab = a + b
                if sab > 128 or sab not in new_S:
                    continue
                return False
        
        # 2. Check triples NOT involving x: {a, b, c} subset of current_S
        # A triple becomes invalid if x is one of its sums, and the other two are in current_S.
        # So, find pairs a,b in current_S such that a+b == x.
        # If found, for each c in current_S \ {a,b}, check if a+c in current_S and b+c in current_S.
        
        # Optimization: Only check if x <= 128
        if x > 128:
            return True
            
        # Find pairs summing to x
        # a < b
        # a >= 1
        for a in range(1, x//2 + 1):
            b = x - a
            if b <= a: continue
            if a in current_S and b in current_S:
                # Check if there exists c in current_S \ {a,b}
                # such that a+c in current_S and b+c in current_S
                for c in current_S:
                    if c == a or c == b: continue
                    if (a + c) in current_S and (b + c) in current_S:
                        # The sums are x, a+c, b+c.
                        # x is being added, so it will be in new_S.
                        # a+c and b+c are in current_S, so in new_S.
                        # Violation!
                        return False
                        
        return True

    rng = random.Random(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        order = candidates[:]
        rng.shuffle(order)
        
        current_S = base_S.copy()
        for x in order:
            if is_valid_addition(x, current_S):
                current_S.add(x)
        
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_set = current_S.copy()
            
    return list(best_set)
