import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds out of 90
    
    N = 180
    
    # Precompute all forbidden triples
    # a = kxy, b = kx(x+y), c = ky(x+y)
    # a, b, c distinct
    # x < y, gcd(x, y) = 1
    triples = []
    for k in range(1, N + 1):
        for x in range(1, N + 1):
            for y in range(x + 1, N + 1):
                if (x + y) > N: # c = k*y*(x+y) >= k*y > k*(x+1) ... 
                    # Actually c can be large, but if c > N, it's not in [1, N]
                    # We only care about triples where a, b, c are all <= N
                    pass
                
                # Check bounds
                a = k * x * y
                if a > N:
                    break # y increasing, so a increases
                b = k * x * (x + y)
                c = k * y * (x + y)
                
                if b > N or c > N:
                    continue
                    
                # Check distinctness
                if a == b or a == c or b == c:
                    continue
                    
                # Check gcd
                if __import__('math').gcd(x, y) != 1:
                    continue
                    
                triples.append((a, b, c))
                
    # Convert to set of frozensets for fast lookup? 
    # Or just list of lists.
    
    # Greedy construction
    # Start with the upper half
    S = set(range(91, 181))
    
    def is_safe(candidate_set, n):
        # Check if adding n creates a conflict
        # We need to check all triples containing n
        for t in triples:
            if n in t:
                # Check if the other two are in candidate_set
                others = [x for x in t if x != n]
                if others[0] in candidate_set and others[1] in candidate_set:
                    return False
        return True

    # Try to add elements from 1 to 90
    # Order matters? Maybe try smaller ones first, or random.
    # Smaller numbers are part of more triples?
    # Let's try adding in increasing order first.
    
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    
    for n in candidates:
        if time.time() > deadline:
            break
        if is_safe(S, n):
            S.add(n)
            
    # Local search / Hill Climbing
    # Try to swap an element out and another in?
    # Or just try to add more if possible? (Already did)
    # Maybe remove an element to allow two others?
    
    # Let's do a few iterations of random swaps
    # Actually, just greedy addition is often good.
    # Let's try multiple random orders.
    
    best_S = S
    best_size = len(S)
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        S_try = set(range(91, 181))
        candidates = list(range(1, 91))
        random.shuffle(candidates)
        
        for n in candidates:
            if time.time() > deadline:
                break
            if is_safe(S_try, n):
                S_try.add(n)
                
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_S = S_try
            
    return list(best_S)
