import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin
    
    def generate_random_graph(n, p=0.5):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def has_regular_induced_subgraph(adj, n, min_size=6):
        # Precompute all subsets of size >= min_size
        # To save time, we can generate on the fly and check in increasing order of size
        # But precomputing might be faster if we do it once for a given n
        # Since n is fixed per call, we can precompute
        subsets_by_size = {}
        for k in range(min_size, n+1):
            subsets_by_size[k] = list(combinations(range(n), k))
        
        # Check in increasing order of size
        for k in range(min_size, n+1):
            for subset in subsets_by_size[k]:
                # Compute degrees in the induced subgraph
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degrees = []
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    degrees.append(deg)
                if len(set(degrees)) == 1:
                    return True
        return False
    
    # Try n=21
    for _ in range(10000):  # Max attempts
        if time.time() > deadline:
            break
        n = 21
        adj = generate_random_graph(n, 0.5)
        if not has_regular_induced_subgraph(adj, n, 6):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Try n=20
    for _ in range(10000):
        if time.time() > deadline:
            break
        n = 20
        adj = generate_random_graph(n, 0.5)
        if not has_regular_induced_subgraph(adj, n, 6):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a small graph
    return {'n': 5, 'edges': []}
