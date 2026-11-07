import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    # Build adjacency set for fast lookup
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Bron-Kerbosch with pivot to find maximum clique
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot u from P U X that maximizes |P ∩ N(u)|
        u = None
        max_neighbors = -1
        for v in list(P) + list(X):
            if len(P & adj[v]) > max_neighbors:
                max_neighbors = len(P & adj[v])
                u = v
        
        # Recurse on candidates not adjacent to u
        candidates = list(P - adj[u])
        for v in candidates:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(P['n'])), set())
    
    # If no clique found (shouldn't happen), return a single edge
    if not max_clique:
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    branches = max_clique
    # Paths are direct edges for a clique
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u = branches[i]
            v = branches[j]
            # Since it's a clique, u and v are adjacent
            paths.append([u, v])
            
    return {'branches': branches, 'paths': paths}

if __name__ == '__main__':
    start_time = time.time()
    result = construct()
    print(f"Time taken: {time.time() - start_time:.4f}s")
    print(f"Number of branches: {len(result['branches'])}")
