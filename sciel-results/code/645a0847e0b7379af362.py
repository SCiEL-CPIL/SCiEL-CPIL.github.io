"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Track remaining edges as a set of frozensets
    remaining = set(frozenset((u, v)) for u, v in edges)
    
    bicliques = []
    
    def find_biclique(remaining_edges):
        """Find a large biclique within the remaining edges."""
        if not remaining_edges:
            return None
        
        # Pick a vertex with max degree in remaining graph
        deg = {}
        for e in remaining_edges:
            u, v = tuple(e)
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        if not deg:
            return None
        
        # Try each vertex as a potential left-side anchor
        best_biclique = None
        best_size = 0
        
        for anchor in deg:
            # Neighbors of anchor in remaining graph
            neighbors = set()
            for e in remaining_edges:
                u, v = tuple(e)
                if u == anchor:
                    neighbors.add(v)
                elif v == anchor:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            # Greedily grow biclique: left={anchor}, right=neighbors
            # Then try to add more left vertices that connect to all right vertices
            left = {anchor}
            right = set(neighbors)
            
            # Try to expand left side
            for v in list(neighbors):
                # Check if v connects to all vertices in right
                v_adj = set()
                for e in remaining_edges:
                    u, w = tuple(e)
                    if u == v:
                        v_adj.add(w)
                    elif w == v:
                        v_adj.add(u)
                
                if right.issubset(v_adj):
                    left.add(v)
                    right.discard(v)  # v moves from right to left
                    # Re-check: all left must connect to all right
                    valid = True
                    for lv in left:
                        lv_adj = set()
                        for e in remaining_edges:
                            u, w = tuple(e)
                            if u == lv:
                                lv_adj.add(w)
                            elif w == lv:
                                lv_adj.add(u)
                        if not right.issubset(lv_adj):
                            valid = False
                            break
                    if not valid:
                        left.discard(v)
                        right.add(v)
            
            # Now try to expand right side with vertices that connect to all left
            for v in range(n):
                if v in left:
                    continue
                v_adj = set()
                for e in remaining_edges:
                    u, w = tuple(e)
                    if u == v:
                        v_adj.add(w)
                    elif w == v:
                        v_adj.add(u)
                
                if left.issubset(v_adj):
                    right.add(v)
            
            size = len(left) * len(right)
            if size > best_size and len(left) > 0 and len(right) > 0:
                best_size = size
                best_biclique = (left, right)
        
        return best_biclique
    
    while remaining and (time.time() - t0) < deadline:
        bc = find_biclique(remaining)
        if bc is None:
            # Fallback: single edge bicliques
            for e in list(remaining):
                u, v = tuple(e)
                bicliques.append([[u], [v]])
                remaining.discard(e)
            break
        
        left, right = bc
        # Verify all cross pairs are in remaining
        valid = True
        for u in left:
            for v in right:
                if frozenset((u, v)) not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or len(left) == 0 or len(right) == 0:
            # Fallback
            for e in list(remaining):
                u, v = tuple(e)
                bicliques.append([[u], [v]])
                remaining.discard(e)
            break
        
        # Remove these edges
        for u in left:
            for v in right:
                remaining.discard(frozenset((u, v)))
        
        bicliques.append([sorted(left), sorted(right)])
    
    # Final fallback for any remaining edges
    for e in remaining:
        u, v = tuple(e)
        bicliques.append([[u], [v]])
    
    # Ensure proper types: list of [list, list]
    result = []
    for bc in bicliques:
        left = [int(x) for x in bc[0]]
        right = [int(x) for x in bc[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
