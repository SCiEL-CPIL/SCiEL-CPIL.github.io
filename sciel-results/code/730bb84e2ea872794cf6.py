import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 180
    
    # Precompute all triples (x, y, z) with x < y < z such that 1/x = 1/y + 1/z
    # 1/x = (y+z)/(yz) => yz = x(y+z) => yz - xy - xz = 0 => (y-x)(z-x) = x^2
    # Let d = y-x, e = z-x. Then de = x^2.
    # y = x + d, z = x + e.
    # We need y, z <= N and y < z => d < e.
    
    # triples_involving[a] = list of (b, c) where {a,b,c} is a triple.
    # For a check "can I add a?", I need to see if there's a pair (b,c) in triples_involving[a] 
    # such that b in S and c in S.
    
    triples_involving = {i: [] for i in range(1, N + 1)}
    
    for x in range(1, N + 1):
        x2 = x * x
        # Find factors d, e of x^2 with d < e
        for d in range(1, int(x2**0.5) + 1):
            if x2 % d != 0:
                continue
            e = x2 // d
            if d >= e:
                continue
            y = x + d
            z = x + e
            if y <= N and z <= N:
                # Triple is (x, y, z)
                triples_involving[x].append((y, z))
                triples_involving[y].append((x, z))
                triples_involving[z].append((x, y))
                
    best_set = set()
    best_size = 0
    
    elements = list(range(1, N + 1))
    
    # Run multiple randomized greedy passes
    while time.time() < deadline:
        random.shuffle(elements)
        S = set()
        
        for a in elements:
            # Check if adding a creates a conflict
            # A conflict exists if there is a pair (b, c) in triples_involving[a]
            # such that b in S and c in S.
            can_add = True
            for b, c in triples_involving[a]:
                if b in S and c in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(a)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
