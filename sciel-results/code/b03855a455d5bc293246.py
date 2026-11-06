import random
import time

def construct():
    def is_valid(n, adj):
        # adj is a list of int bitmasks
        # Check all subsets of size >= 6
        # Optimization: Iterate masks
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check if induced subgraph on 'mask' is regular
            # Get degrees of vertices in mask
            first_deg = -1
            is_reg = True
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                d = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return False
        
        return True

    def generate_random_regular(n, d, seed):
        random.seed(seed)
        # Configuration model
        # List of (vertex, stub)
        stubs = []
        for i in range(n):
            for _ in range(d):
                stubs.append(i)
        
        # Shuffle
        random.shuffle(stubs)
        
        # Pair up
        adj = [0] * n
        has_multi = False
        has_self = False
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                has_self = True
                break
            if (adj[u] >> v) & 1:
                has_multi = True
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        if has_self or has_multi:
            return None
            
        # Verify it is regular
        for i in range(n):
            if adj[i].bit_count() != d:
                return None
                
        return adj

    # Try to find a valid graph on n=21
    # Regular graphs with degree 5 or 6 are good candidates
    start_time = time.time()
    max_time = 100.0
    
    # Try different degrees
    for d in [5, 6, 7]:
        if time.time() - start_time > max_time:
            break
        for seed in range(500):
            if time.time() - start_time > max_time:
                break
            adj = generate_random_regular(21, d, seed)
            if adj is not None:
                if is_valid(21, adj):
                    edges = []
                    for i in range(21):
                        for j in range(i+1, 21):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': 21, 'edges': edges}
    
    # Fallback to n=20
    for d in [5, 6, 7]:
        if time.time() - start_time > 115.0:
            break
        for seed in range(500):
            if time.time() - start_time > 115.0:
                break
            adj = generate_random_regular(20, d, seed)
            if adj is not None:
                if is_valid(20, adj):
                    edges = []
                    for i in range(20):
                        for j in range(i+1, 20):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': 20, 'edges': edges}
                    
    # Fallback to random G(n,p) if regular graphs fail
    for p in [0.4, 0.5, 0.6]:
        if time.time() - start_time > 118.0:
            break
        for seed in range(100):
            if time.time() - start_time > 118.0:
                break
            random.seed(seed)
            adj = [0] * 20
            for i in range(20):
                for j in range(i+1, 20):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if is_valid(20, adj):
                edges = []
                for i in range(20):
                    for j in range(i+1, 20):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}

    return {'n': 5, 'edges': []}
