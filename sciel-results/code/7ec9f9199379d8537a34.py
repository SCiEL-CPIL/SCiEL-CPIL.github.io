import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def check_c4_free(adj_mask, n):
        """Check if graph is C4-free using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                # If more than 1 bit is set, there is a C4
                if common & (common - 1):
                    return False
        return True

    def compute_alpha(adj_mask, n):
        """Compute maximum independent set size using Branch and Bound."""
        best = [0]
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            # Upper bound: current size + number of remaining candidates
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: Pick vertex with max degree in the candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses a greedy random construction with backtracking/repair.
        """
        if time.time() > deadline:
            return None
            
        # We want to construct a graph on n vertices.
        # Greedy approach:
        # 1. Initialize empty graph.
        # 2. For each vertex u, try to add edges to future vertices v > u.
        # 3. Constraint: For any u, v, they can share at most 1 neighbor.
        #    This means if we add edge (u, w), we must ensure that for all existing neighbors v of u,
        #    w is not already a neighbor of v (unless v=w, which is impossible for simple graph).
        #    Actually, the condition is: for any pair (u, v), |N(u) intersection N(v)| <= 1.
        #    
        #    When adding edge (u, w) where w > u:
        #    - For any v < u that is a neighbor of u:
        #      If v is also a neighbor of w, then u and v share neighbor w.
        #      They must not share any other neighbor.
        #      Let common = N(u) & N(v).
        #      If w is in common, and common already has size 1, we can't add (u, w) if w is the 2nd common neighbor?
        #      No, if w is in N(v), and w is added to N(u), then w becomes a common neighbor.
        #      If u and v already have a common neighbor x, then adding (u, w) where w is also in N(v) would make 2 common neighbors (x and w).
        #      So: if w is in N(v), and (N(u) & N(v)) is non-empty, we cannot add (u, w).
        #      If (N(u) & N(v)) is empty, adding (u, w) makes w a common neighbor (size 1), which is allowed.
        #      
        #    - For any v > u that is a neighbor of w (future? no, we process u in order, so w is not processed yet? 
        #      Wait, if we iterate u from 0 to n-1, and for each u we decide edges to v > u,
        #      then when we are at u, N(w) for w > u is not fully determined yet?
        #      Actually, if we only add edges (u, v) with u < v, then when processing u, 
        #      the neighbors of w (w > u) are only those x < w that have already connected to w.
        #      Since u < w, if we add (u, w), u becomes a neighbor of w.
        #      Does this violate C4-free property for any pair (x, w) where x < w?
        #      Yes. We need to check for all x < w, x != u, that are neighbors of w:
        #      If x is a neighbor of u, then u and x share neighbor w.
        #      If u and x already share a neighbor, we cannot add (u, w).
        #
        #    So the check for adding edge (u, w) (u < w) is:
        #    For all v in N(w) (current neighbors of w, which are < w):
        #       If v is in N(u), then u and v share w.
        #       Check if u and v already share a neighbor.
        #       If yes, adding (u, w) creates a C4.
        #
        #    Also, we need to check for pairs (u, v) where v < u?
        #    If we add (u, w), does it affect pairs (u, v) for v < u?
        #    u and v share neighbors. w is a new neighbor of u.
        #    If w is also a neighbor of v, then u and v share w.
        #    Check if u and v already share a neighbor.
        #    So the check is symmetric:
        #    For all v in N(u) U N(w):
        #       If v is in N(u) AND v is in N(w):
        #           Check if u and w share any other neighbor? No, we are adding (u, w).
        #           The condition is: for any pair (a, b), |N(a) & N(b)| <= 1.
        #           Adding (u, w) increases |N(u) & N(v)| by 1 for any v in N(w) \ N(u).
        #           And increases |N(w) & N(v)| by 1 for any v in N(u) \ N(w).
        #           
        #    Simplified check:
        #    Let C_u = N(u)
        #    Let C_w = N(w)
        #    If we add (u, w):
        #    For any v in C_u:
        #       If v in C_w:
        #          u and v share w.
        #          Check if |N(u) & N(v)| >= 1. If yes, adding w makes it >= 2. FAIL.
        #    For any v in C_w:
        #       If v in C_u: (same as above)
        #       If v not in C_u:
        #          w and v share u.
        #          Check if |N(w) & N(v)| >= 1. If yes, adding u makes it >= 2. FAIL.
        
        # This is getting complex. A simpler heuristic is to just try random edges and check global C4-free.
        # But that's slow.
        
        # Let's use the "Greedy Random" approach mentioned in the inspiration:
        # Build graph vertex by vertex.
        # For vertex i, try to connect to previous vertices j < i.
        # Constraint: For any two previous neighbors j1, j2 of i, they must not share any neighbor other than i.
        # Actually, if i connects to j1 and j2, then j1 and j2 share neighbor i.
        # They must not share any other neighbor.
        # So: if j1 and j2 already share a neighbor, we cannot connect i to both.
        
        adj_mask = [0] * n
        
        for i in range(n):
            # Try to connect i to some previous vertices
            # We want to maximize degree to keep alpha low.
            # But we must respect C4-free.
            
            # Candidates: previous vertices j < i
            # We can choose a subset of candidates to connect to i.
            # If we connect i to j1 and j2, then j1 and j2 must not share any neighbor.
            
            # Let's try to pick as many as possible.
            # Greedy: iterate j from i-1 down to 0.
            # If we can connect i to j without violating C4, do it.
            
            # To check if we can connect i to j:
            # 1. For all existing neighbors k of i (k < i):
            #    If k is also a neighbor of j, then i and j share k? No, i and j are being connected.
            #    The new edge is (i, j).
            #    This creates a common neighbor for (i, k) if k is neighbor of j? No.
            #    It creates a common neighbor for (j, k) if k is neighbor of i? Yes, i is neighbor of k and j.
            #    So j and k share neighbor i.
            #    Check if j and k already share a neighbor. If yes, FAIL.
            # 2. Also, does adding (i, j) create a C4 involving i and some other vertex?
            #    i and j share no neighbors yet (since i is new, it has no neighbors other than those we added).
            #    Wait, if we add edges one by one, we must check against all previously added edges.
            
            # Let's maintain a list of neighbors for each vertex.
            # When considering adding edge (i, j):
            # For every neighbor k of i:
            #    If k is a neighbor of j:
            #       Then i and j are both neighbors of k.
            #       Do i and j share any other neighbor? No, we are adding the first edge between them.
            #       But wait, the C4 condition is on pairs of vertices having common neighbors.
            #       If we add (i, j), then for any pair (a, b), the set of common neighbors might change.
            #       Specifically, i and j become neighbors.
            #       This affects pairs (i, v) and (j, v).
            #       For pair (i, v): common neighbors are N(i) & N(v).
            #       If v is a neighbor of j, then j is a common neighbor of i and v? No, j is a neighbor of i and v.
            #       So j is in N(i) & N(v).
            #       We need |N(i) & N(v)| <= 1.
            #       If v is a neighbor of j, and v is NOT a neighbor of i (yet), then adding (i, j) makes j a common neighbor of i and v.
            #       If i and v already share a neighbor, FAIL.
            #       If v is a neighbor of i AND v is a neighbor of j, then v is in N(i) & N(j).
            #       This doesn't directly violate |N(i) & N(v)| <= 1 unless i and v share another neighbor.
            #       But it DOES violate the condition for the pair (i, v)? No.
            #       It violates the condition for the pair (i, v) if |N(i) & N(v)| > 1.
            #       Adding (i, j) adds j to N(i). It does not change N(v).
            #       So if v was in N(j), then j is in N(v).
            #       So j is in N(i) & N(v).
            #       If there was already some x in N(i) & N(v), then we now have {j, x}, size 2. FAIL.
            
            # So the check for adding (i, j) is:
            # For all v in N(j):
            #    If v in N(i):
            #       # v is a common neighbor of i and j? No, v is in N(i) and N(j).
            #       # This means i and j share neighbor v.
            #       # This is allowed as long as they don't share another one.
            #       # But we are adding (i, j).
            #       # Does this affect any other pair?
            #       # Consider pair (i, v). Common neighbors: N(i) & N(v).
            #       # j is in N(i) (new) and j is in N(v) (since v in N(j) and graph is undirected).
            #       # So j is a common neighbor of i and v.
            #       # Check if |N(i) & N(v)| >= 1 before adding?
            #       # If yes, adding j makes it >= 2. FAIL.
            #    Else (v not in N(i)):
            #       # j is in N(i) (new) and j is in N(v).
            #       # So j is a common neighbor of i and v.
            #       # Check if |N(i) & N(v)| >= 1 before adding?
            #       # If yes, adding j makes it >= 2. FAIL.
            
            # Summary: For adding (i, j), for all v in N(j):
            #    Check if |N(i) & N(v)| >= 1. If yes, FAIL.
            #    Wait, is that correct?
            #    If v in N(j), then j in N(v).
            #    We are adding (i, j), so j in N(i).
            #    So j is in N(i) & N(v).
            #    If there is any other common neighbor, size becomes >= 2.
            #    So yes, we must ensure N(i) & N(v) is empty for all v in N(j).
            
            # Additionally, we must ensure that i and j don't already share a neighbor?
            # If they do, say x, then |N(i) & N(j)| >= 1.
            # Adding (i, j) doesn't change N(i) & N(j) directly, but it makes i and j neighbors.
            # The condition is on common neighbors of ANY pair.
            # Pair (i, j): common neighbors are N(i) & N(j).
            # If they already share x, then size is >= 1.
            # Adding (i, j) doesn't add to N(i) & N(j).
            # So it's fine if they share 1 neighbor.
            # But if they share 2 neighbors, it's already invalid.
            
            # So the check is:
            # 1. For all v in N(j):
            #    If (N(i) & N(v)) != 0: FAIL.
            # 2. (Implicitly) The graph must remain C4-free.
            
            # Let's implement this check.
            
            # We want to maximize the number of edges incident to i.
            # So we try to connect i to as many j < i as possible.
            # We can do this greedily.
            
            # To make it more robust, we can try multiple random orders or use a heuristic.
            # For now, let's just try to connect to all possible j in a random order.
            
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                # Check if we can add (i, j)
                can_add = True
                # N(i) is adj_mask[i]
                # N(j) is adj_mask[j]
                
                # For all v in N(j):
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    
                    # Check if N(i) & N(v) is non-empty
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    # Add edge (i, j)
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    # If we have enough edges, we might stop?
                    # We want to maximize n, so we should try to make the graph as dense as possible.
                    # But we are building for a fixed n.
                    # We should try to add as many edges as possible.
        
        # After construction, check if alpha < 11
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Start from high n and go down? Or start from low and go up?
    # High n is harder to satisfy alpha < 11.
    # Let's try n=43, 42, ..., 31.
    # If we find one, return it.
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random constructions
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            adj_mask = try_find_graph(n, 10)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
