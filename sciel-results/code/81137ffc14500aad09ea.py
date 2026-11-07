import random

def construct():
    best_S = set()
    best_size = 0
    
    for _ in range(500):
        S = set()
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            
            # Check 1: x is member c. a,b in S. a+b in S.
            for a in S:
                ax = a + x
                if ax in S:
                    for b in S:
                        if b == a:
                            continue
                        if (a + b in S) and (b + x in S):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Check 2: x is sum a+b. a,b in S. c in S.
                for a in S:
                    b = x - a
                    if b in S and b != a:
                        for c in S:
                            if c == a or c == b:
                                continue
                            if (a + c in S) and (b + c in S):
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                # Check 3: x is member c AND x is sum a+b. a,b in S.
                for a in S:
                    b = x - a
                    if b in S and b != a:
                        if (a + x in S) and (b + x in S):
                            valid = False
                            break
            
            if valid:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return [int(x) for x in best_S]
