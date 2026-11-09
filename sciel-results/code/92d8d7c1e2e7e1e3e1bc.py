import time
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build original adjacency masks
    adj_orig = [0] * n
    for u, v in edges:
        adj_orig[u] |= (1 << v)
        adj_orig[v] |= (1 << u)
        
    # Track uncovered edges using a set of frozensets or a bitmask per vertex
    # For efficiency, we use a bitmask for each vertex representing its uncovered neighbors
    # But we need to ensure symmetry.
    uncovered = [0] * n
    for u, v in edges:
        uncovered[u] |= (1 << v)
        uncovered[v] |= (1 << u)
        
    def get_uncovered_edges_mask():
        # Returns a list of (u, v) tuples for all uncovered edges with u < v
        res = []
        for u in range(n):
            m = uncovered[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    res.append((u, v))
                m ^= lsb
        return res

    def count_uncovered_edges():
        count = 0
        for u in range(n):
            count += bin(uncovered[u]).count('1')
        return count // 2

    # Generate candidate bicliques from the original graph
    # Strategy: For each edge (u, v), find the maximal biclique containing it.
    # To speed up, we can just generate bicliques based on common neighbors.
    # A simple heuristic candidate: For each vertex u, let R = N(u). 
    # This is a star, which is a biclique [ {u}, N(u) ].
    # Better: For each pair (u, v) in edges, let L = {u, v} initially? No.
    
    # Let's generate candidates by expanding from each edge.
    candidates = []
    seen_candidates = set()
    
    # To keep it fast, we limit the number of candidates or the expansion depth.
    # For n=30, we can afford to generate a decent number.
    # Let's generate maximal bicliques for a subset of "seed" edges.
    # Seed edges: All edges.
    
    def expand_biclique(L_mask, R_mask):
        # Expand L and R to be maximal w.r.t original graph
        changed = True
        while changed:
            changed = False
            # Find vertices that can be added to L
            # A vertex w can be added to L if it is connected to all in R
            # And w is not in R
            # And w is not in L
            # Actually, we want a biclique in the original graph.
            # So w must be connected to all v in R.
            # And we must ensure no edge is missing.
            
            # Compute intersection of neighborhoods of all v in R
            # If R is empty, any vertex can be added to L? 
            # But L and R must be non-empty.
            
            if R_mask == 0:
                break
                
            common_R = (1 << n) - 1
            m = R_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                common_R &= adj_orig[v]
                m ^= lsb
                
            # Candidates for L: vertices in common_R, not in R, not in L
            # We can add any subset. To maximize, we add all?
            # If we add all, it might break the property that L vertices are connected to R?
            # No, if w is in common_R, it is connected to all in R.
            # So adding w to L is valid.
            # What about adding to R?
            common_L = (1 << n) - 1
            m = L_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                common_L &= adj_orig[u]
                m ^= lsb
                
            # New L candidates: common_R & ~R_mask & ~L_mask
            new_L_cands = common_R & ~R_mask & ~L_mask
            # New R candidates: common_L & ~L_mask & ~R_mask
            new_R_cands = common_L & ~L_mask & ~R_mask
            
            # We can add all of them? 
            # If we add w to L, it must be connected to all in R. (Satisfied by common_R)
            # If we add z to R, it must be connected to all in L. (Satisfied by common_L)
            # But if we add w to L, does it affect R? 
            # w must be connected to all existing R. Yes.
            # Does w need to be connected to new R candidates? 
            # If we add z to R, z must be connected to all L, including w.
            # Is z connected to w? 
            # z is in common_L (old L). w is in new_L_cands.
            # We don't know if z-w is an edge.
            # So we cannot just add all. We need to be careful.
            
            # Simple expansion: Add one vertex at a time? Or just find the maximal one via BFS?
            # For now, let's just return the current L, R if no simple expansion works.
            # Or, let's just use the "Star" bicliques and "Edge" bicliques as candidates.
            # This is safe and fast.
            
            # Let's try a simple greedy expansion:
            # Add vertex to L if it connects to all R.
            # Add vertex to R if it connects to all L.
            # Repeat until no change.
            
            if new_L_cands:
                # Add one? Or all?
                # Adding all might be invalid if they don't connect to new R.
                # Let's just add the one with highest degree in common_R?
                # To keep it simple and fast, let's just stop here and use the initial seed.
                pass
            if new_R_cands:
                pass
                
        return L_mask, R_mask

    # The expansion above is too complex/buggy for a quick rewrite.
    # Let's use a simpler candidate generation:
    # 1. All stars: [ [u], N(u) ] for each u.
    # 2. All edges: [ [u], [v] ] for each edge (u,v).
    # 3. For each vertex u, and each neighbor v, let L={u}, R=N(u). 
    #    Then try to expand L by adding neighbors of all in R?
    
    # Let's stick to a robust "Greedy on Uncovered" but with a different seed selection.
    # Instead of max degree, use "Max Common Neighborhood".
    # For each edge (u,v) in uncovered, compute |N(u) & N(v)|.
    # Pick the edge with the largest common neighborhood.
    # Build biclique around it.
    
    bicliques = []
    start_time = time.time()
    deadline = start_time + 60
    
    while count_uncovered_edges() > 0:
        if time.time() > deadline:
            break
            
        # Find best seed edge
        best_edge = None
        best_common_size = -1
        
        for u in range(n):
            m = uncovered[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    # Calculate common neighbors in uncovered graph? 
                    # Or in original graph?
                    # We want a biclique that covers many uncovered edges.
                    # So common neighbors should be from the uncovered graph?
                    # If w is a common neighbor of u and v in the uncovered graph,
                    # then (u,w) and (v,w) are uncovered.
                    # Also (u,v) is uncovered.
                    # So {u, w} x {v, w}? No, {u, v} x {w}? 
                    # If we pick L={u, v}, R={w}, edges are (u,w), (v,w).
                    # If we pick L={u}, R={v, w}, edges are (u,v), (u,w).
                    
                    # Let's estimate coverage.
                    # Common neighbors in uncovered:
                    common = uncovered[u] & uncovered[v]
                    # Exclude u and v
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    size = bin(common).count('1')
                    
                    if size > best_common_size:
                        best_common_size = size
                        best_edge = (u, v)
                m ^= lsb
        
        if best_edge is None:
            # Fallback: pick any edge
            for u in range(n):
                m = uncovered[u]
                if m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    best_edge = (u, v)
                    break
        
        if best_edge is None:
            break
            
        u0, v0 = best_edge
        
        # Build biclique starting with L={u0}, R={v0}
        L_mask = (1 << u0)
        R_mask = (1 << v0)
        
        # Expand greedily
        # We want to maximize the number of uncovered edges covered.
        # An edge (x, y) is covered if x in L, y in R.
        # Currently covered: (u0, v0).
        
        # Candidate additions to L: neighbors of all in R (in uncovered graph)
        # Candidate additions to R: neighbors of all in L (in uncovered graph)
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            
            # Find candidates for L
            # Must be connected to all in R via uncovered edges
            common_R = (1 << n) - 1
            m = R_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                common_R &= uncovered[v]
                m ^= lsb
            
            # Candidates for L: in common_R, not in L, not in R
            cand_L = common_R & ~L_mask & ~R_mask
            
            # Find candidates for R
            common_L = (1 << n) - 1
            m = L_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                common_L &= uncovered[u]
                m ^= lsb
                
            cand_R = common_L & ~L_mask & ~R_mask
            
            # Select best candidate to add
            # Gain of adding w to L: number of new edges = |R| (since w connected to all R)
            # But wait, adding w to L creates edges (w, v) for all v in R.
            # Are these edges uncovered? Yes, by definition of cand_L.
            # So gain is |R|.
            
            # Gain of adding z to R: number of new edges = |L|.
            
            gain_L = bin(R_mask).count('1')
            gain_R = bin(L_mask).count('1')
            
            best_gain = -1
            best_add = None # (type, vertex)
            
            if cand_L:
                # Pick any? Or one that allows more future expansion?
                # For now, pick first one.
                lsb = cand_L & -cand_L
                w = lsb.bit_length() - 1
                if gain_L > best_gain:
                    best_gain = gain_L
                    best_add = ('L', w)
            
            if cand_R:
                lsb = cand_R & -cand_R
                z = lsb.bit_length() - 1
                if gain_R > best_gain:
                    best_gain = gain_R
                    best_add = ('R', z)
            
            if best_add:
                typ, vtx = best_add
                if typ == 'L':
                    L_mask |= (1 << vtx)
                else:
                    R_mask |= (1 << vtx)
                changed = True
        
        L_list = [i for i in range(n) if (L_mask & (1 << i))]
        R_list = [i for i in range(n) if (R_mask & (1 << i))]
        
        if not L_list or not R_list:
            # Should not happen if we started with an edge
            break
            
        bicliques.append([L_list, R_list])
        
        # Remove covered edges from uncovered
        for u in L_list:
            for v in R_list:
                if (uncovered[u] & (1 << v)):
                    uncovered[u] &= ~(1 << v)
                    uncovered[v] &= ~(1 << u)

    # Convert to required format
    final_result = []
    for bic in bicliques:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
        
    return final_result
