# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 100
    n = 16
    
    # Q3 edges in canonical form (8 vertices, 12 edges)
    q3e = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Q3 adjacency
    q3_adj = {v: [] for v in range(8)}
    for a, b in q3e:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
    
    # Helper to get edge index
    def eidx(u, v):
        if u > v: u, v = v, u
        return u * 16 + v
        
    def creates_q3(mask, new_edge):
        """Check if adding new_edge to mask creates a Q3 subgraph."""
        u, v = new_edge
        # For each Q3 edge, try both orientations
        for a, b in q3e:
            for perm in [(a, b), (b, a)]:
                pa, pb = perm
                assign = {pa: u, pb: v}
                used = {u, v}
                remaining = [i for i in range(8) if i not in (pa, pb)]
                
                def bt(idx):
                    if time.time() > deadline:
                        return True # Assume conflict to be safe
                    if idx == len(remaining):
                        for ea, eb in q3e:
                            x, y = assign[ea], assign[eb]
                            if (x, y) == new_edge or (y, x) == new_edge:
                                continue
                            if not (mask >> eidx(x, y)) & 1:
                                return False
                        return True
                    qv = remaining[idx]
                    for cand in range(n):
                        if cand in used:
                            continue
                        ok = True
                        for nb in q3_adj[qv]:
                            if nb in assign:
                                x = assign[nb]
                                if (cand, x) == new_edge or (x, cand) == new_edge:
                                    continue
                                if not (mask >> eidx(cand, x)) & 1:
                                    ok = False
                                    break
                        if ok:
                            assign[qv] = cand
                            used.add(cand)
                            res = bt(idx + 1)
                            if res:
                                return True
                            del assign[qv]
                            used.remove(cand)
                    return False
                
                if bt(0):
                    return True
        return False

    def find_q3_subgraph(mask):
        """Find a Q3 subgraph in the graph defined by mask. Returns list of edges or None."""
        # Try all subsets of 8 vertices? 16 choose 8 = 12870.
        # For each, check if induced subgraph is Q3.
        # Or use backtracking similar to creates_q3 but without a "new_edge" constraint, just finding any.
        
        # A faster heuristic: Pick a random edge, try to extend to Q3.
        # But for correctness, we should be thorough if we rely on it for validation.
        # Given n=16, checking all 8-sets is feasible if the check is fast.
        
        # Let's use a backtracking search for any Q3.
        # Fix vertex 0 to be vertex 0 of Q3.
        # This reduces symmetry.
        
        # Actually, the `creates_q3` logic is adaptable.
        # We can just try to find a mapping of Q3 vertices to graph vertices.
        
        def bt_q3(idx, assign, used):
            if time.time() > deadline:
                return None
            if idx == 8:
                # Verify all edges
                for ea, eb in q3e:
                    x, y = assign[ea], assign[eb]
                    if not (mask >> eidx(x, y)) & 1:
                        return None
                return [(assign[ea], assign[eb]) for ea, eb in q3e]
            
            qv = idx
            for cand in range(n):
                if cand in used:
                    continue
                ok = True
                for nb in q3_adj[qv]:
                    if nb < 8 and nb in assign: # Only check assigned neighbors
                        x = assign[nb]
                        if not (mask >> eidx(cand, x)) & 1:
                            ok = False
                            break
                if ok:
                    assign[qv] = cand
                    used.add(cand)
                    res = bt_q3(idx + 1, assign, used)
                    if res:
                        return res
                    del assign[qv]
                    used.remove(cand)
            return None
        
        # To avoid symmetry breaking issues, we can fix vertex 0 of Q3 to vertex 0 of Graph?
        # No, Q3 might not involve vertex 0.
        # We can iterate over all vertices for the first Q3 vertex.
        # But 16 * (backtracking) might be slow.
        
        # Alternative: Just use the `creates_q3` function on a "virtual" edge?
        # No, creates_q3 checks if adding an edge *creates* a Q3.
        
        # Let's just do a simple check: 
        # For every set of 8 vertices, check if it induces Q3.
        # 12870 sets.
        # Check: 12 edges.
        # 12870 * 12 = 154k operations. Very fast.
        
        from itertools import combinations
        all_vertices = list(range(n))
        for combo in combinations(all_vertices, 8):
            # Map combo to 0..7
            mapping = {v: i for i, v in enumerate(combo)}
            inv_mapping = {i: v for i, v in enumerate(combo)}
            
            # Check if edges of Q3 exist in graph
            is_q3 = True
            for a, b in q3e:
                u, v = inv_mapping[a], inv_mapping[b]
                if not (mask >> eidx(u, v)) & 1:
                    is_q3 = False
                    break
            
            if is_q3:
                # Ensure no extra edges? No, subgraph isomorphic means edges exist.
                # It doesn't require induced subgraph to be exactly Q3, just that Q3 is a subgraph.
                # So if all 12 edges are present, it's a Q3 subgraph.
                return [(inv_mapping[a], inv_mapping[b]) for a, b in q3e]
        
        return None

    # Initialize with Complement of Shrikhande Graph
    # Shrikhande graph edges:
    # Vertices 0..15.
    # One representation:
    # The Shrikhande graph is the graph with vertices (i,j) for i,j in 0..3
    # Edges if:
    # 1. i same, j adjacent in C4? No.
    # 2. Standard construction:
    # It is a strongly regular graph (16,6,2,2).
    # Let's use a known edge list or generate it.
    # The Shrikhande graph is the complement of the 4x4 Rook graph? No.
    # The Rook graph (Lattice graph L_2(K_4)) has vertices (i,j). Edges if i=i' or j=j'.
    # Degree of Rook: 3 (same row) + 3 (same col) = 6.
    # So Rook is (16,6,2,2).
    # Shrikhande is the other (16,6,2,2).
    # So Shrikhande is NOT the Rook graph.
    # The complement of Rook is (16,9,4,4).
    # The complement of Shrikhande is (16,9,4,4).
    
    # We want to start with a dense graph.
    # Let's start with the Complement of the Rook Graph (which is dense, 72 edges).
    # Is it Q3-free?
    # If not, we remove edges.
    
    # Rook Graph Edges:
    rook_edges = set()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if j == k: continue
                # Same row
                rook_edges.add((i*4+j, i*4+k))
                if i == k: continue
                # Same col
                rook_edges.add((i*4+j, k*4+j))
    
    # Complement of Rook
    edges = set()
    edge_mask = 0
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in rook_edges:
                edges.add((u, v))
                edge_mask |= (1 << eidx(u, v))
                
    # Check if initial graph is Q3-free
    q3_found = find_q3_subgraph(edge_mask)
    while q3_found:
        # Remove one edge from the found Q3
        # Randomly pick one to break it
        random.seed(int(time.time() * 1000) % 10000)
        edge_to_remove = random.choice(q3_found)
        u, v = edge_to_remove
        edges.discard((u, v))
        edges.discard((v, u))
        edge_mask &= ~(1 << eidx(u, v))
        
        # Re-check
        q3_found = find_q3_subgraph(edge_mask)
        if time.time() > deadline - 20:
            break
            
    # Now we have a Q3-free graph.
    # Run the existing greedy add and local search to maximize edges.
    
    all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    missing = list(all_edges - edges)
    random.seed(42)
    random.shuffle(missing)
    
    for e in missing:
        if time.time() > deadline - 15:
            break
        if creates_q3(edge_mask, e):
            continue
        edges.add(e)
        edge_mask |= (1 << eidx(*e))
        
    # Local Search: 1-for-k swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_edges = list(edges)
        random.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() > deadline - 5:
                break
            
            mask_without_e = edge_mask & ~(1 << eidx(*e))
            temp_mask = mask_without_e
            temp_added = []
            
            for m in all_edges:
                if m == e:
                    continue
                if (temp_mask >> eidx(*m)) & 1:
                    continue
                if creates_q3(temp_mask, m):
                    continue
                temp_mask |= (1 << eidx(*m))
                temp_added.append(m)
                
            if len(temp_added) > 1:
                edges.discard(e)
                for m in temp_added:
                    edges.add(m)
                edge_mask = temp_mask
                improved = True
                break
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
