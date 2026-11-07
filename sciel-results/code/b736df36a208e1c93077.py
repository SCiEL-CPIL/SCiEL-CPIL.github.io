"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import random
import sys
from collections import defaultdict

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set((min(a,b), max(a,b)) for a,b in edges)
    
    # Build adjacency matrix for O(1) lookup
    adj_matrix = [[False] * n for _ in range(n)]
    for a, b in edges:
        adj_matrix[a][b] = True
        adj_matrix[b][a] = True
        
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # Helper to check if a set of edges forms a valid biclique
    # Returns (L, R) if valid, else None
    def get_biclique_structure(edge_subset):
        if not edge_subset:
            return None
        # Build subgraph
        sub_adj = defaultdict(set)
        for u, v in edge_subset:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
            
        # Check if bipartite and get components
        # Since it's a biclique, it should be one component (if connected) or multiple K_{1,1}?
        # A biclique K_{L,R} is connected if L,R non-empty.
        # If edge_subset is empty, handled.
        
        # Bipartite check
        color = {}
        for node in sub_adj:
            if node not in color:
                # BFS
                queue = [node]
                color[node] = 0
                while queue:
                    curr = queue.pop(0)
                    for neighbor in sub_adj[curr]:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[curr]
                            queue.append(neighbor)
                        else:
                            if color[neighbor] == color[curr]:
                                return None # Not bipartite
        
        # Determine L and R
        L = set()
        R = set()
        for node in sub_adj:
            if color[node] == 0:
                L.add(node)
            else:
                R.add(node)
                
        # Check if all edges between L and R in G are in edge_subset
        # And no edges within L or within R in edge_subset (guaranteed by bipartite)
        for u in L:
            for v in R:
                if adj_matrix[u][v]:
                    if (min(u,v), max(u,v)) not in edge_subset:
                        return None
                else:
                    # If not an edge in G, it's fine, it's just not in the biclique
                    # But wait, the biclique is defined by the edges in G.
                    # If (u,v) is not in G, it cannot be in the biclique.
                    # The condition "Every cross pair must be a host edge" means
                    # if u in L and v in R, (u,v) MUST be an edge in G.
                    # So if adj_matrix[u][v] is False, this is NOT a valid biclique
                    # because the definition requires ALL cross pairs to be edges.
                    return None
                    
        return (L, R)

    # Initial Greedy Partition
    uncovered = set(edge_set)
    bicliques = []
    
    while uncovered:
        best_biclique_edges = None
        best_size = 0
        
        # Sample a few random edges to seed bicliques to diversify
        # Or just iterate all? For N=30, E~200, iterating all is fine.
        for e in list(uncovered):
            u, v = e
            # Expand to maximal biclique containing e
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in L and w not in R:
                        if all(adj_matrix[w][x] for x in R):
                            L.add(w)
                            changed = True
                for w in range(n):
                    if w not in L and w not in R:
                        if all(adj_matrix[w][x] for x in L):
                            R.add(w)
                            changed = True
            
            # Verify and get edge set
            # Check if all cross pairs are in uncovered
            valid = True
            for a in L:
                for b in R:
                    if (min(a,b), max(a,b)) not in uncovered:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    # Store the edges of this biclique
                    biclique_edges = set()
                    for a in L:
                        for b in R:
                            biclique_edges.add((min(a,b), max(a,b)))
                    best_biclique_edges = biclique_edges
        
        if best_biclique_edges is None:
            break
            
        bicliques.append(best_biclique_edges)
        uncovered.difference_update(best_biclique_edges)
        
    # Local Search: Merge bicliques
    # Represent bicliques by their edge sets
    # We store them as sets of edges.
    # We need to convert back to (L,R) for final output.
    
    changed = True
    while changed:
        changed = False
        n_bicliques = len(bicliques)
        for i in range(n_bicliques):
            if i >= len(bicliques): continue
            for j in range(i+1, len(bicliques)):
                if j >= len(bicliques): continue
                # Try to merge bicliques[i] and bicliques[j]
                union_edges = bicliques[i] | bicliques[j]
                L_new, R_new = None, None
                # Check if union is a valid biclique
                # Use the helper
                res = get_biclique_structure(union_edges)
                if res is not None:
                    L_new, R_new = res
                    # Merge
                    bicliques.pop(j)
                    bicliques[i] = union_edges
                    changed = True
                    break
            if changed:
                break

    # Final validation and conversion
    result = []
    for b_edges in bicliques:
        L, R = get_biclique_structure(b_edges)
        if L is None:
            # Should not happen, but fallback to individual edges if broken
            # This would invalidate the partition if we just skip, so we must ensure validity
            # Actually, if get_biclique_structure fails, it's an error in our logic.
            # We will assume it works.
            pass
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
        
    # Verify partition
    covered = set()
    for bic in result:
        L, R = bic
        for u in L:
            for v in R:
                covered.add((min(u,v), max(u,v)))
                
    if covered != edge_set:
        # Fallback to simple edge-by-edge partition if something went wrong
        result = []
        for e in edge_set:
            result.append([[e[0]], [e[1]]])
            
    return result

# EVOLVE-BLOCK-END
