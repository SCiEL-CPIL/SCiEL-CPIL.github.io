import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy: repeatedly find largest biclique in remaining edges
    remaining = set(edge_set)
    bicliques = []
    
    def find_max_biclique(rem_edges):
        """Find a large complete bipartite subgraph in rem_edges."""
        if not rem_edges:
            return None, None
        
        # Get vertices involved
        verts = set()
        for u, v in rem_edges:
            verts.add(u)
            verts.add(v)
        
        # Try to find a good biclique
        best_L = None
        best_R = None
        best_size = 0
        
        # For each vertex, compute its neighbors in remaining graph
        # Group vertices by their neighbor sets to find potential biclique sides
        
        # Strategy: pick a vertex with high degree in remaining, 
        # then find common neighbors
        for u in verts:
            # Neighbors of u in remaining
            nbrs_u = set()
            for v in adj[u]:
                e = (min(u,v), max(u,v))
                if e in rem_edges:
                    nbrs_u.add(v)
            
            if not nbrs_u:
                continue
            
            # Try to find a subset of nbrs_u that all connect to a common set
            # Start with u on left, try to maximize right side
            # For biclique [L, R] where u in L, R must be subset of nbrs_u
            # and all vertices in L must connect to all in R
            
            # Simple: just use u and its neighbors as a star
            # But we can do better: find max clique in the "common neighbor" structure
            
            # For now, try: L = {u}, R = nbrs_u
            size = len(nbrs_u)
            if size > best_size:
                best_size = size
                best_L = [u]
                best_R = sorted(nbrs_u)
            
            # Try to expand: find vertices that connect to all of nbrs_u
            # or find a subset of nbrs_u and vertices connecting to that subset
            
        # Also try finding bicliques where both sides have multiple vertices
        # Use a heuristic: for pairs of vertices, check if they share neighbors
        for u in verts:
            nbrs_u = set()
            for v in adj[u]:
                e = (min(u,v), max(u,v))
                if e in rem_edges:
                    nbrs_u.add(v)
            if len(nbrs_u) < 2:
                continue
            
            # Try to find R subset of nbrs_u and L containing u
            # Start with R = nbrs_u, L = {u}
            # Try to add more vertices to L that connect to all of R
            R = set(nbrs_u)
            L = {u}
            for w in verts:
                if w == u or w in R:
                    continue
                # Check if w connects to all of R
                all_conn = True
                for r in R:
                    e = (min(w,r), max(w,r))
                    if e not in rem_edges:
                        all_conn = False
                        break
                if all_conn:
                    L.add(w)
            
            # Now try to add more to R that connect to all of L
            changed = True
            while changed:
                changed = False
                for w in verts:
                    if w in L or w in R:
                        continue
                    all_conn = True
                    for l in L:
                        e = (min(w,l), max(w,l))
                        if e not in rem_edges:
                            all_conn = False
                            break
                    if all_conn:
                        R.add(w)
                        changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_L = sorted(L)
                best_R = sorted(R)
        
        if best_L is None or best_R is None:
            # Fallback: pick any edge
            u, v = next(iter(rem_edges))
            return [u], [v]
        
        return best_L, best_R
    
    # Greedy extraction
    while remaining:
        if time.time() > deadline:
            break
        
        L, R = find_max_biclique(remaining)
        if L is None:
            # Extract individual edges
            for e in list(remaining):
                u, v = e
                bicliques.append([[u], [v]])
            remaining.clear()
            break
        
        # Verify all cross pairs are in remaining
        valid = True
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: extract single edge
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((min(u,v), max(u,v)))
            continue
        
        bicliques.append([L, R])
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                remaining.discard(e)
    
    # If we timed out, fill remaining edges as single-edge bicliques
    if remaining:
        for e in remaining:
            u, v = e
            bicliques.append([[u], [v]])
    
    # Local improvement: try to merge bicliques
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, len(bicliques)):
                    if time.time() > deadline - 10:
                        break
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # Try merging into [L1∪L2, R1∪R2] or [L1∪R2, R1∪L2] etc.
                    # For simplicity, try [L1+L2, R1+R2]
                    Lnew = list(set(L1 + L2))
                    Rnew = list(set(R1 + R2))
                    if not Lnew or not Rnew:
                        continue
                    # Check all cross pairs are edges
                    valid = True
                    for u in Lnew:
                        for v in Rnew:
                            e = (min(u,v), max(u,v))
                            if e not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Check no self-loops (u != v for all cross pairs)
                        valid = True
                        for u in Lnew:
                            for v in Rnew:
                                if u == v:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        # Verify the merged biclique's edges are exactly the union of the two
                        merged_edges = set()
                        for u in Lnew:
                            for v in Rnew:
                                merged_edges.add((min(u,v), max(u,v)))
                        orig_edges = set()
                        for u in L1:
                            for v in R1:
                                orig_edges.add((min(u,v), max(u,v)))
                        for u in L2:
                            for v in R2:
                                orig_edges.add((min(u,v), max(u,v)))
                        if merged_edges == orig_edges:
                            # Merge
                            bicliques[i] = [Lnew, Rnew]
                            bicliques.pop(j)
                            improved = True
                            break
                if improved:
                    break
    
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L_clean = [int(x) for x in L]
        R_clean = [int(x) for x in R]
        if L_clean and R_clean:
            result.append([L_clean, R_clean])
    
    return result
# EVOLVE-BLOCK-END
