import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Fallback popcount
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    # Try n from 20 down to 5
    for n in range(20, 4, -1):
        if time.time() > deadline:
            break
        
        # For each n, try a few random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            adj = [0] * n
            # Use p=0.5
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # We will try to eliminate all regular induced subgraphs of size >= 6
            for step in range(500):
                if time.time() > deadline:
                    break
                
                violation = None
                viol_k = 0
                
                # Check k=6
                # To speed up, we can pre-compute the list of 6-combinations?
                # itertools.combinations is C-implemented, so it's fast.
                for subset in itertools.combinations(range(n), 6):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    
                    first_deg = -1
                    regular = True
                    for v in subset:
                        d = popcount(adj[v] & mask)
                        if first_deg == -1:
                            first_deg = d
                        elif d != first_deg:
                            regular = False
                            break
                    if regular:
                        violation = subset
                        viol_k = 6
                        break
                
                if violation is None:
                    # Check k=7
                    for subset in itertools.combinations(range(n), 7):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        first_deg = -1
                        regular = True
                        for v in subset:
                            d = popcount(adj[v] & mask)
                            if first_deg == -1:
                                first_deg = d
                            elif d != first_deg:
                                regular = False
                                break
                        if regular:
                            violation = subset
                            viol_k = 7
                            break
                
                if violation is None:
                    # Check k=8..n
                    for k in range(8, n+1):
                        for subset in itertools.combinations(range(n), k):
                            mask = 0
                            for v in subset:
                                mask |= (1 << v)
                            first_deg = -1
                            regular = True
                            for v in subset:
                                d = popcount(adj[v] & mask)
                                if first_deg == -1:
                                    first_deg = d
                                elif d != first_deg:
                                    regular = False
                                    break
                            if regular:
                                violation = subset
                                viol_k = k
                                break
                        if violation is not None:
                            break
                
                if violation is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Fix violation
                # Flip a random edge inside the violation
                u, v = random.sample(violation, 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # End local search
        # End attempts for this n
        # If we didn't find a valid graph for this n, try smaller n
        # Note: If we fail on 20, we move to 19.
        
    return {'n': 5, 'edges': []}
