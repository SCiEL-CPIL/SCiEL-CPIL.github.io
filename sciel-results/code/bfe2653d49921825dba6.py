import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Start with K_{8,8}, prune to Q3-free, then add intra-partition edges
    # Partitions: X = {0..7}, Y = {8..15}
    X = list(range(8))
    Y = list(range(8, 16))
    
    # Start with complete bipartite graph
    edges = set()
    for x in X:
        for y in Y:
            edges.add((x, y))
    
    # Adjacency masks
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute permutations
    perms24 = list(itertools.permutations(range(4)))
    idx4 = list(itertools.combinations(range(8), 4))
    
    # Fast Q3 check using bitmask operations
    def contains_q3_fast(verts):
        """Check if 8 vertices contain Q3 as a subgraph."""
        v_list = list(verts)
        
        for (i, j, k, l) in idx4:
            A = [v_list[i], v_list[j], v_list[k], v_list[l]]
            B = [v_list[m] for m in range(8) if m not in (i, j, k, l)]
            
            # Precompute adjacency masks for A and B
            a_masks = [adj_mask[a] for a in A]
            b_masks = [adj_mask[b] for b in B]
            
            # Orientation 1: A=even parity, B=odd parity
            # Required edges for permutation p:
            # a0 -> B[p[0]], B[p[1]], B[p[3]]
            # a1 -> B[p[0]], B[p[1]], B[p[2]]
            # a2 -> B[p[0]], B[p[2]], B[p[3]]
            # a3 -> B[p[1]], B[p[2]], B[p[3]]
            for p in perms24:
                # Check a0
                m0 = a_masks[0]
                if not ((m0 >> B[p[0]]) & 1): continue
                if not ((m0 >> B[p[1]]) & 1): continue
                if not ((m0 >> B[p[3]]) & 1): continue
                # Check a1
                m1 = a_masks[1]
                if not ((m1 >> B[p[0]]) & 1): continue
                if not ((m1 >> B[p[1]]) & 1): continue
                if not ((m1 >> B[p[2]]) & 1): continue
                # Check a2
                m2 = a_masks[2]
                if not ((m2 >> B[p[0]]) & 1): continue
                if not ((m2 >> B[p[2]]) & 1): continue
                if not ((m2 >> B[p[3]]) & 1): continue
                # Check a3
                m3 = a_masks[3]
                if not ((m3 >> B[p[1]]) & 1): continue
                if not ((m3 >> B[p[2]]) & 1): continue
                if not ((m3 >> B[p[3]]) & 1): continue
                return True
            
            # Orientation 2: A=odd parity, B=even parity
            for p in perms24:
                # b0 -> A[p[0]], A[p[1]], A[p[3]]
                m0 = b_masks[0]
                if not ((m0 >> A[p[0]]) & 1): continue
                if not ((m0 >> A[p[1]]) & 1): continue
                if not ((m0 >> A[p[3]]) & 1): continue
                # b1 -> A[p[0]], A[p[1]], A[p[2]]
                m1 = b_masks[1]
                if not ((m1 >> A[p[0]]) & 1): continue
                if not ((m1 >> A[p[1]]) & 1): continue
                if not ((m1 >> A[p[2]]) & 1): continue
                # b2 -> A[p[0]], A[p[2]], A[p[3]]
                m2 = b_masks[2]
                if not ((m2 >> A[p[0]]) & 1): continue
                if not ((m2 >> A[p[2]]) & 1): continue
                if not ((m2 >> A[p[3]]) & 1): continue
                # b3 -> A[p[1]], A[p[2]], A[p[3]]
                m3 = b_masks[3]
                if not ((m3 >> A[p[1]]) & 1): continue
                if not ((m3 >> A[p[2]]) & 1): continue
                if not ((m3 >> A[p[3]]) & 1): continue
                return True
        return False
    
    def is_q3_free():
        """Check if the entire graph is Q3-free."""
        all_verts = list(range(16))
        for combo in itertools.combinations(all_verts, 8):
            if contains_q3_fast(combo):
                return False
        return True
    
    def add_edge_safe(u, v):
        """Add edge (u,v) if it doesn't create a Q3."""
        # Temporarily add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check all 8-sets containing both u and v
        remaining = [x for x in range(16) if x != u and x != v]
        for combo in itertools.combinations(remaining, 6):
            verts = [u, v] + list(combo)
            if contains_q3_fast(verts):
                # Remove and return False
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        return True
    
    # Phase 1: Prune K_{8,8} to make it Q3-free
    # Remove edges from bipartite graph until Q3-free
    bipartite_edges = [(x, y) for x in X for y in Y]
    
    # Sort by a heuristic: remove edges that are "most critical" first
    # For now, try removing in a systematic order
    removed_count = 0
    for e in bipartite_edges:
        if time.time() > deadline - 5:
            break
        u, v = e
        if (u, v) in edges:
            # Check if removing this edge helps
            # Actually, let's check if the current graph is Q3-free
            # If not, we need to remove more edges
            pass
    
    # Better approach: iteratively check and remove
    # First, check if K_{8,8} is Q3-free (it definitely isn't)
    # We need to remove enough edges
    
    # Use a greedy removal strategy:
    # While graph contains Q3, find a Q3 and remove one of its edges
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Check if Q3-free
        q3_found = None
        all_verts = list(range(16))
        for combo in itertools.combinations(all_verts, 8):
            if contains_q3_fast(combo):
                q3_found = combo
                break
        
        if q3_found is None:
            break
        
        # Found a Q3, remove one edge from it
        # Choose the edge to remove: pick one that minimizes impact
        # For simplicity, remove the first edge we can
        q3_verts = list(q3_found)
        # Find all edges within this 8-set that are in the graph
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = q3_verts[i], q3_verts[j]
                if (u, v) in edges:
                    q3_edges.append((u, v))
        
        if q3_edges:
            # Remove one edge
            # Heuristic: remove an edge that appears in fewer potential Q3s
            # For now, just remove the first one
            e_to_remove = q3_edges[0]
            edges.remove(e_to_remove)
            adj_mask[e_to_remove[0]] &= ~(1 << e_to_remove[1])
            adj_mask[e_to_remove[1]] &= ~(1 << e_to_remove[0])
            removed_count += 1
        else:
            break
    
    # Phase 2: Add intra-partition edges
    # Try adding edges within X and within Y
    intra_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            intra_edges.append((X[i], X[j]))
            intra_edges.append((Y[i], Y[j]))
    
    # Sort by degree sum (higher first)
    def degree(v):
        return bin(adj_mask[v]).count('1')
    
    intra_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
    
    # Try to add each intra-partition edge
    for e in intra_edges:
        if time.time() > deadline - 5:
            break
        u, v = e
        if (u, v) not in edges:
            if add_edge_safe(u, v):
                pass  # Edge added
    
    # Phase 3: Try to add more bipartite edges that were removed
    bipartite_non_edges = [(x, y) for x in X for y in Y if (x, y) not in edges]
    bipartite_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
    
    for e in bipartite_non_edges:
        if time.time() > deadline - 3:
            break
        u, v = e
        if add_edge_safe(u, v):
            pass  # Edge added
    
    return [list(e) for e in edges]
