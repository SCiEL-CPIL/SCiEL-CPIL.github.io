"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges_list = P['edges']
    edge_set = set()
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    # Build adjacency as sets for fast intersection
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    
    remaining = set(edge_set)
    bicliques = []
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    while remaining:
        if time.time() > deadline:
            # Fallback: individual edges
            for e in remaining:
                bicliques.append([[int(e[0])], [int(e[1])]])
            break
        
        best_biclique = None
        best_count = 0
        
        # Try expanding from a limited set of seed edges
        # Prioritize edges in dense regions
        seed_edges = list(remaining)
        # Sort by sum of degrees in remaining graph
        rem_deg = [0] * n
        for a, b in remaining:
            rem_deg[a] += 1
            rem_deg[b] += 1
        seed_edges.sort(key=lambda e: rem_deg[e[0]] + rem_deg[e[1]], reverse=True)
        
        max_seeds = min(len(seed_edges), 30)
        
        for seed in seed_edges[:max_seeds]:
            u, v = seed
            L = {u}
            R = {v}
            
            # Iteratively expand to maximal biclique
            changed = True
            while changed:
                changed = False
                # Try to add to L: x must connect to all in R
                for x in range(n):
                    if x in L or x in R:
                        continue
                    if all((min(x, y), max(x, y)) in remaining for y in R):
                        L.add(x)
                        changed = True
                # Try to add to R: y must connect to all in L
                for y in range(n):
                    if y in L or y in R:
                        continue
                    if all((min(x, y), max(x, y)) in remaining for x in L):
                        R.add(y)
                        changed = True
            
            # Count edges in this biclique
            count = 0
            valid = True
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    if e in remaining:
                        count += 1
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and count > best_count:
                best_count = count
                best_biclique = (list(L), list(R))
        
        if best_biclique is None or best_count == 0:
            # Single edge fallback
            e = next(iter(remaining))
            bicliques.append([[int(e[0])], [int(e[1])]])
            remaining.discard(e)
        else:
            bicliques.append(best_biclique)
            L, R = best_biclique
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    remaining.discard(e)
    
    # Ensure correct types
    result = []
    for b in bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
