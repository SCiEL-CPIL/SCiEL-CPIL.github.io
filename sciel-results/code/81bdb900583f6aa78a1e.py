import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = list(edge_set)
    
    # Helper: get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Helper: check if (L_mask, R_mask) is a valid biclique covering only edges in 'remaining_mask'
    # Returns the count of edges covered, or 0 if invalid
    def get_valid_biclique_count(L_mask, R_mask, remaining_mask):
        if L_mask == 0 or R_mask == 0:
            return 0
        
        # Check 1: All cross pairs must be edges in the original graph
        # For each l in L, adj_mask[l] must contain R_mask
        lm = L_mask
        while lm:
            low = lm & (-lm)
            v = low.bit_length() - 1
            if (adj_mask[v] & R_mask) != R_mask:
                return 0
            lm ^= low
            
        # Check 2: The cross edges must be a subset of remaining_mask
        # We need to check if every (l, r) pair is in remaining_mask
        # Efficient way: Iterate through L, for each l, check if R_mask is a subset of 
        # (adj_mask[l] & remaining_mask) ... wait, remaining_mask is a set of edges, not vertices.
        
        # Since n is small (30), we can just iterate and check.
        # But we can optimize: 
        # For a fixed l, the edges to R are determined.
        # Let's just iterate.
        count = 0
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            rm = R_mask
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                e = (min(u, v), max(u, v))
                if e in remaining_mask:
                    count += 1
                else:
                    # If an edge is not in remaining, the biclique is not valid for the current step
                    # because we would be covering an edge already covered or not existing.
                    # Actually, the condition is "cross-edge sets must partition all pinned graph edges".
                    # So we can only pick edges that are in `remaining`.
                    # If (u,v) is an edge in the graph but not in `remaining`, we cannot include it in this biclique's coverage.
                    # However, the definition of a biclique (L, R) implies ALL cross pairs are edges.
                    # So if (u,v) is a cross pair, it MUST be covered by this biclique.
                    # If it's not in `remaining`, we have a conflict.
                    return 0
                rm ^= low_r
            lm ^= low
            
        return count

    def expand_biclique(start_edge, remaining_mask):
        u, v = start_edge
        L_mask = 1 << u
        R_mask = 1 << v
        
        # We want to maximize the number of edges covered from `remaining_mask`
        # while maintaining the biclique property and ensuring all covered edges are in `remaining_mask`.
        
        # Strategy: Greedily add vertices that increase the valid edge count.
        # A vertex x can be added to L if:
        # 1. x is not in L or R.
        # 2. For all r in R, (x, r) is an edge in the graph.
        # 3. For all r in R, (x, r) is in `remaining_mask`.
        
        # Similarly for R.
        
        improved = True
        while improved:
            improved = False
            
            # Try adding to L
            for x in range(n):
                if x == u or x == v:
                    continue
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                
                # Check if x can join L
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    # Must be neighbor
                    if not (adj_mask[x] >> r) & 1:
                        ok = False
                        break
                    # Must be in remaining
                    e = (min(x, r), max(x, r))
                    if e not in remaining_mask:
                        ok = False
                        break
                    rm ^= low_r
                
                if ok:
                    L_mask |= (1 << x)
                    improved = True
            
            # Try adding to R
            for y in range(n):
                if y == u or y == v:
                    continue
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue
                
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not (adj_mask[y] >> l) & 1:
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining_mask:
                        ok = False
                        break
                    lm ^= low_l
                
                if ok:
                    R_mask |= (1 << y)
                    improved = True
                    
        return L_mask, R_mask

    def greedy_init(seed_order):
        remaining = set(all_edges)
        bicliques = []
        
        # Sort edges by some heuristic or just use the provided order
        # We shuffle the order of picking edges to get different restarts
        edge_list = list(remaining)
        # Apply seed_order permutation if needed, or just shuffle
        # For simplicity, we shuffle the list of edges to pick from
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            best_L = 0
            best_R = 0
            best_count = 0
            
            # Sample a few edges to find a good biclique
            # Instead of all, sample up to 20 random edges from remaining
            sample_size = min(20, len(remaining))
            sample = random.sample(list(remaining), sample_size)
            
            for e in sample:
                L_mask, R_mask = expand_biclique(e, remaining)
                count = get_valid_biclique_count(L_mask, R_mask, remaining)
                if count > best_count:
                    best_count = count
                    best_L = L_mask
                    best_R = R_mask
            
            if best_L == 0:
                # Fallback: pick any edge
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
                best_count = 1 # At least the starting edge
                
            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])
            
            # Remove covered edges from remaining
            # We need to remove all (l, r) pairs
            for u in L_list:
                for v in R_list:
                    remaining.discard((min(u, v), max(u, v)))
                    
        return bicliques

    def try_merge(bicliques, i, j):
        """
        Try to merge biclique i and j.
        Returns a new list of bicliques if successful, else None.
        """
        Li, Ri = bicliques[i]
        Lj, Rj = bicliques[j]
        
        # Get edge sets
        ei = set()
        for u in Li:
            for v in Ri:
                ei.add((min(u, v), max(u, v)))
                
        ej = set()
        for u in Lj:
            for v in Rj:
                ej.add((min(u, v), max(u, v)))
                
        union_edges = ei | ej
        
        # Get vertices involved in union_edges
        vertices = set()
        for u, v in union_edges:
            vertices.add(u)
            vertices.add(v)
            
        # Check if the subgraph induced by union_edges is a complete bipartite graph
        # 1. Bipartite check
        color = {}
        # Build local adjacency for the subgraph
        sub_adj = {v: set() for v in vertices}
        for u, v in union_edges:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
            
        for start in vertices:
            if start not in color:
                color[start] = 0
                stack = [start]
                while stack:
                    curr = stack.pop()
                    for nb in sub_adj[curr]:
                        if nb not in color:
                            color[nb] = 1 - color[curr]
                            stack.append(nb)
                        elif color[nb] == color[curr]:
                            return None # Odd cycle, not bipartite
        
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # 2. Completeness check
        # All pairs in L x R must be in union_edges
        L_set = set(L)
        R_set = set(R)
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in union_edges:
                    return None
                    
        # 3. Validity check against original graph
        # All pairs in L x R must be edges in the original graph
        for u in L:
            for v in R:
                if not (adj_mask[u] >> v) & 1:
                    return None
        
        # If valid, construct new biclique list
        new_bicliques = []
        for k, b in enumerate(bicliques):
            if k != i and k != j:
                new_bicliques.append(b)
        new_bicliques.append([L, R])
        
        return new_bicliques

    def local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    result = try_merge(bicliques, i, j)
                    if result is not None:
                        bicliques = result
                        improved = True
                        break
                if improved:
                    break
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        bicliques = greedy_init(restart)
        
        if not bicliques:
            continue
            
        bicliques = local_search(bicliques)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
            # If we reach a very low number, we can stop early?
            # The minimum is 1. If we get 1, we can break.
            if best_count == 1:
                break

    if best_bicliques is None:
        # Fallback: single edges
        best_bicliques = [[[u], [v]] for u, v in all_edges]
        
    # Final conversion to ensure types are correct
    result = []
    for left, right in best_bicliques:
        # Ensure lists of ints
        result.append([list(left), list(right)])
        
    return result

# EVOLVE-BLOCK-END
