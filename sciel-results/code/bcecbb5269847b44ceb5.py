import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build edge set and adjacency bitmask
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # adj_mask[i] = bitmask of neighbors of i in the ORIGINAL graph
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # uncovered_mask[i] = bitmask of uncovered neighbors of i
    uncovered_mask = [0] * n
    for u, v in edges:
        uncovered_mask[u] |= (1 << v)
        uncovered_mask[v] |= (1 << u)
    
    def popcount(x):
        return bin(x).count('1')
    
    def bits_of(mask):
        """Return list of vertex indices set in mask."""
        result = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            result.append(idx)
            m ^= lb
        return result
    
    def valid_biclique(L, R, unc_mask):
        """Check if (L,R) forms a valid biclique in the uncovered graph."""
        if not L or not R:
            return False
        # Check L is independent in uncovered graph
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                if unc_mask[L[i]] & (1 << L[j]):
                    return False
        # Check R is independent in uncovered graph
        for i in range(len(R)):
            for j in range(i+1, len(R)):
                if unc_mask[R[i]] & (1 << R[j]):
                    return False
        # Check all cross edges exist in uncovered graph
        for u in L:
            for v in R:
                if not (unc_mask[u] & (1 << v)):
                    return False
        return True
    
    def expand_biclique(L, R, unc_mask):
        """Try to add vertices to L or R to maximize covered edges."""
        L = list(L)
        R = list(R)
        improved = True
        while improved:
            improved = False
            # Try adding to L
            for v in range(n):
                if v in L or v in R:
                    continue
                # v can join L if: v has no uncovered edges to L, and v has uncovered edges to all R
                ok = True
                for u in L:
                    if unc_mask[v] & (1 << u):
                        ok = False
                        break
                if not ok:
                    continue
                for w in R:
                    if not (unc_mask[v] & (1 << w)):
                        ok = False
                        break
                if ok and popcount(unc_mask[v] & 0) >= 0:
                    # Check v is not connected to any in R via uncovered (already checked above)
                    # Actually we need: v not connected to L (independent) AND v connected to all R
                    # We already checked both. Add v.
                    L.append(v)
                    improved = True
                    break
            # Try adding to R
            for v in range(n):
                if v in L or v in R:
                    continue
                ok = True
                for u in R:
                    if unc_mask[v] & (1 << u):
                        ok = False
                        break
                if not ok:
                    continue
                for w in L:
                    if not (unc_mask[v] & (1 << w)):
                        ok = False
                        break
                if ok:
                    R.append(v)
                    improved = True
                    break
        return L, R
    
    def find_max_biclique(unc_mask, deadline):
        """Find a large biclique in the uncovered graph using bitmask operations."""
        best_L = None
        best_R = None
        best_covered = 0
        
        # Find vertex with max uncovered degree
        max_deg = 0
        u_star = -1
        for i in range(n):
            d = popcount(unc_mask[i])
            if d > max_deg:
                max_deg = d
                u_star = i
        
        if u_star < 0 or max_deg == 0:
            return None, None, 0
        
        # Candidate: L={u_star}, R=neighbors of u_star
        R = bits_of(unc_mask[u_star])
        covered = len(R)
        if covered > best_covered:
            best_covered = covered
            best_L = [u_star]
            best_R = R
        
        # Try L={u_star, v} for v in neighbors of u_star
        for v in R:
            if time.time() > deadline:
                break
            # Common neighbors of u_star and v in uncovered graph
            common = unc_mask[u_star] & unc_mask[v]
            common &= ~(1 << u_star)
            common &= ~(1 << v)
            # v must not be connected to any in R (but R is just neighbors of u_star, and v is in R)
            # Actually for L={u_star, v}, R must be common neighbors excluding u_star and v
            # But we also need u_star and v to NOT have an uncovered edge between them? 
            # Wait: L must be independent. u_star and v ARE connected (v is neighbor of u_star).
            # So L={u_star, v} is NOT valid if (u_star,v) is an uncovered edge!
            # We need L to be an independent set. So we can't put adjacent vertices in L.
            # Let me reconsider: for a biclique (L,R), L must be independent, R must be independent,
            # and every u in L must be connected to every v in R.
            # So u_star and v can't both be in L if they're adjacent.
            # Instead, try: L={u_star}, and try to expand R, or try different structures.
            
            # Alternative: try R = {v}, L = common neighbors of u_star and v
            # L must be independent. Common neighbors of u_star and v.
            # But we need L to be independent in the uncovered graph.
            
            # Let's try a different approach: for each pair (u_star, v) where v is NOT adjacent to u_star,
            # we can put both in L.
            pass
        
        # Try: for each vertex w NOT adjacent to u_star, L={u_star, w}
        # R = common neighbors of u_star and w
        non_neighbors = []
        for w in range(n):
            if w == u_star:
                continue
            if not (unc_mask[u_star] & (1 << w)):
                non_neighbors.append(w)
        
        for w in non_neighbors:
            if time.time() > deadline:
                break
            common = unc_mask[u_star] & unc_mask[w]
            common &= ~(1 << u_star)
            common &= ~(1 << w)
            R_cand = bits_of(common)
            if not R_cand:
                continue
            # Check R_cand is independent
            valid = True
            for i in range(len(R_cand)):
                for j in range(i+1, len(R_cand)):
                    if unc_mask[R_cand[i]] & (1 << R_cand[j]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            covered = 2 * len(R_cand)
            if covered > best_covered:
                best_covered = covered
                best_L = [u_star, w]
                best_R = R_cand
        
        # Try triples: L={u_star, w, x} where w,x not adjacent to u_star and not to each other
        if len(non_neighbors) >= 2:
            for i in range(len(non_neighbors)):
                if time.time() > deadline:
                    break
                w = non_neighbors[i]
                for j in range(i+1, len(non_neighbors)):
                    if time.time() > deadline:
                        break
                    x = non_neighbors[j]
                    # w and x must not be adjacent
                    if unc_mask[w] & (1 << x):
                        continue
                    common = unc_mask[u_star] & unc_mask[w] & unc_mask[x]
                    common &= ~(1 << u_star)
                    common &= ~(1 << w)
                    common &= ~(1 << x)
                    R_cand = bits_of(common)
                    if len(R_cand) < 2:
                        continue
                    # Check R_cand is independent
                    valid = True
                    for ii in range(len(R_cand)):
                        for jj in range(ii+1, len(R_cand)):
                            if unc_mask[R_cand[ii]] & (1 << R_cand[jj]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    covered = 3 * len(R_cand)
                    if covered > best_covered:
                        best_covered = covered
                        best_L = [u_star, w, x]
                        best_R = R_cand
        
        if best_L is None:
            # Fallback: single edge
            for u in range(n):
                if unc_mask[u]:
                    v = bits_of(unc_mask[u])[0]
                    best_L = [u]
                    best_R = [v]
                    best_covered = 1
                    break
        
        return best_L, best_R, best_covered
    
    # Phase 1: Greedy extraction
    bicliques = []
    while True:
        if time.time() > deadline:
            break
        # Check if any uncovered edges remain
        has_uncovered = False
        for i in range(n):
            if unc_mask[i]:
                has_uncovered = True
                break
        if not has_uncovered:
            break
        
        L, R, covered = find_max_biclique(unc_mask, deadline)
        if L is None:
            break
        
        # Expand
        L, R = expand_biclique(L, R, unc_mask)
        
        # Final validation
        if not valid_biclique(L, R, unc_mask):
            # Fallback to single edge
            u = L[0]
            v = R[0]
            L = [u]
            R = [v]
        
        bicliques.append([L, R])
        
        # Remove covered edges
        for u in L:
            for v in R:
                unc_mask[u] &= ~(1 << v)
                unc_mask[v] &= ~(1 << u)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 5:
        # Try merging pairs of bicliques
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(bicliques)):
                    if time.time() > deadline - 5:
                        break
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try to merge: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                    # But this is only valid if the union forms a biclique.
                    # More practically: try to find a biclique that covers edges from both.
                    # Simple approach: try L1 ∪ L2 as L, and find common neighbors.
                    # Or try various combinations.
                    
                    # Try: L = L1 ∪ L2, R = common neighbors in original graph
                    # But we need to work with the remaining edges, not original.
                    # Actually, after greedy, all edges are covered. To improve, we need to
                    # re-partition. This is complex. Let's try a simpler merge:
                    # If L1 and L2 are disjoint, and R1 and R2 are disjoint, check if
                    # (L1 ∪ L2, R1 ∪ R2) is a valid biclique in the ORIGINAL graph.
                    # If so, we can replace two bicliques with one.
                    
                    # Check: all cross edges in (L1∪L2, R1∪R2) must be in original graph
                    valid = True
                    for u in L1:
                        for v in R2:
                            if not (adj_mask[u] & (1 << v)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if not (adj_mask[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        # Also need L1∪L2 independent and R1∪R2 independent in original graph
                        # (they were independent in their respective bicliques, but cross-edges within L1∪L2)
                        l_union = L1 + L2
                        r_union = R1 + R2
                        # Check L_union independent
                        for a in range(len(l_union)):
                            for b in range(a+1, len(l_union)):
                                if adj_mask[l_union[a]] & (1 << l_union[b]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            for a in range(len(r_union)):
                                for b in range(a+1, len(r_union)):
                                    if adj_mask[r_union[a]] & (1 << r_union[b]):
                                        valid = False
                                        break
                            if not valid:
                                break
                        
                        if valid:
                            # Verify all cross edges
                            for u in l_union:
                                for v in r_union:
                                    if not (adj_mask[u] & (1 << v)):
                                        valid = False
                                        break
                                if not valid:
                                    break
                        
                        if valid:
                            # Replace bicliques i and j with merged one
                            new_biclique = [l_union, r_union]
                            del bicliques[j]
                            del bicliques[i]
                            bicliques.append(new_biclique)
                            improved = True
                            break
                if improved:
                    break
    
    # Convert to required format
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
    
    return result
