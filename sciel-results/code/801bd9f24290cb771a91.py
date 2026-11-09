import time
import itertools

def construct():
    start_time = time.time()
    deadline = 60.0
    
    # Universe: {49, ..., 100}
    # Min sum of 4 elements: 49+50+51+52 = 202 > 200
    # So only subsets of size 1, 2, 3 can sum to 200.
    # Size 1: max 100 < 200.
    # Size 2: max 100+99=199 < 200.
    # Size 3: possible.
    
    U = list(range(49, 101))
    
    # Generate all triples summing to 200
    triples = []
    n = len(U)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if U[i] + U[j] + U[k] == 200:
                    triples.append((U[i], U[j], U[k]))
    
    # Greedy Hitting Set
    unhit = set(triples)
    removals = set()
    
    while unhit:
        if time.time() > deadline - 5:
            break
            
        # Count frequencies
        freq = {}
        for t in unhit:
            for x in t:
                freq[x] = freq.get(x, 0) + 1
        
        if not freq:
            break
            
        # Pick element with max frequency
        max_elem = max(freq, key=lambda x: freq[x])
        removals.add(max_elem)
        
        # Remove all triples containing max_elem
        unhit = {t for t in unhit if max_elem not in t}
        
    result = [x for x in U if x not in removals]
    
    # Fallback to seed-like behavior if result is empty or too small (should not happen)
    if not result:
        # Fallback to simple greedy from seed
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
        
    return result
