import random
import time
from itertools import combinations

def construct():
    def generate_random_regular(n, d, seed):
        if n * d % 2 != 0:
            return None
        if d >= n:
            return None
        random.seed(seed)
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(d):
                half_edges.append(i)
        random.shuffle(half_edges)
        
        # Pair up half edges
        adj = [0] * n
        has_self_loop = False
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                has_self_loop = True
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if has_self_loop:
            return None
            
        # Verify simplicity and regularity
        for i in range(n):
            if adj[i].bit_count() != d:
                return None
            if adj[i] & (1 << i):
                return None
        return adj

    def is_valid(n, adj):
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                # Create mask for this subset
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if induced subgraph is regular
                first_deg = -1
                is_reg = True
                for v in comb:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    # Try to find a valid graph on n=21
    start_time = time.time()
    target_time_21 = 15.0  # Spend up to 15s on n=21
    
    # Degrees 5 and 6 are good candidates for n=21
    for d in [5, 6]:
        for seed in range(500):
            if time.time() - start_time > target_time_21:
                break
            adj = generate_random_regular(21, d, seed * 100 + d)
            if adj is None:
                continue
            if is_valid(21, adj):
                edges = []
                for i in range(21):
                    for j in range(i + 1, 21):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': 21, 'edges': edges}
    
    # Fallback to n=20
    for d in [5, 6]:
        for seed in range(500):
            if time.time() - start_time > 110.0:
                break
            adj = generate_random_regular(20, d, seed * 100 + d + 10)
            if adj is None:
                continue
            if is_valid(20, adj):
                edges = []
                for i in range(20):
                    for j in range(i + 1, 20):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}
                
    # If all else fails, return a trivially valid small graph
    return {'n': 5, 'edges': []}
