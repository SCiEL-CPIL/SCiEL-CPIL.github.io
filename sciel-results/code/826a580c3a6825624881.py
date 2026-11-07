import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def solve(edge_set, seed_val):
        rng = random.Random(seed_val)
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            # Find best biclique: try all pairs (u,v) as same-side seeds
            best_cover = 0
            best_L = None
            best_R = None
            
            # Try each vertex as single-left
            for u in range(n):
                R = [v for v in adj[u] if (min(u,v), max(u,v)) in remaining]
                if R:
                    cover = len(R)
                    if cover > best_cover:
                        best_cover = cover
                        best_L = [u]
                        best_R = R
            
            # Try pairs (u,v) as left side
            # Only try if it could beat current best
            if best_cover < 10:  # only bother if current best is small
                for u in range(n):
                    for v in adj[u]:
                        if u >= v:
                            continue
                        # Common neighbors
                        common = adj[u] & adj[v]
                        R = [w for w in common if w != u and w != v 
                            and (min(u,w),max(u,w)) in remaining 
                            and (min(v,w),max(v,w)) in remaining]
                        if len(R) >= 2:
                            cover = 2 * len(R)
                            if cover > best_cover:
                                best_cover = cover
                                best_L = [u, v]
                                best_R = R
            
            if best_L is None:
                # Shouldn't happen
                break
            
            bicliques.append([best_L, best_R])
            # Remove covered edges
            for x in best_L:
                for y in best_R:
                    if x != y:
                        remaining.discard((min(x,y), max(x,y)))
        
        return bicliques
    
    best = None
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 55
    
    for seed in range(300):
        if time.time() > deadline:
            break
        result = solve(edge_set, seed)
        if len(result) < best_count:
            best_count = len(result)
            best = result
    
    if best is None:
        # Fallback
        best = []
        remaining = set(edge_set)
        for u in range(n):
            R = [v for v in adj[u] if (min(u,v), max(u,v)) in remaining]
            if R:
                best.append([[u], R])
                for v in R:
                    remaining.discard((min(u,v), max(u,v)))
    
    final = []
    for bic in best:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        final.append([L, R])
    
    return final
