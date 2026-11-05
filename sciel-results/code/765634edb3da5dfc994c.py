import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_count = 0
    best_adj = [[False] * n for _ in range(n)]
    
    # Q3 structure
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Precompute all 8! / 2 = 20160 embeddings of Q3 into 16 vertices
    # Actually, we need to check if adding edge (u,v) creates a Q3.
    # A Q3 contains 12 edges. If we add edge (u,v), we need to find a Q3 where
    # (u,v) is one of the 12 edges and all other 11 edges are in the current graph.
    
    # Better approach: precompute all possible Q3 embeddings (injective maps from Q3 vertices to graph vertices)
    # There are 16*15*14*13*12*11*10*9 = 16!/8! = 518,918,400 such maps, too many.
    # But we only need to check if a specific edge (u,v) can be part of a Q3.
    
    # For each pair (u,v) and each of the 24 orientations of Q3 edges,
    # check if the remaining 6 Q3 vertices can be embedded.
    
    def creates_q3_fast(u, v, adj_bits):
        """Check if adding edge (u,v) to graph with adjacency bitmasks creates a Q3."""
        # Try all 24 orientations (12 edges * 2 directions)
        for i, j in q3_edges:
            # Case 1: u->i, v->j
            if _check_embedding_fast(i, j, u, v, adj_bits, q3_adj, n):
                return True
            # Case 2: u->j, v->i
            if _check_embedding_fast(j, i, u, v, adj_bits, q3_adj, n):
                return True
        return False
    
    def _check_embedding_fast(q3_u, q3_v, g_u, g_v, adj_bits, q3_adj, n):
        """
        Check if there's an embedding of Q3 where q3_u->g_u, q3_v->g_v,
        and the edge (g_u, g_v) is the new edge being added.
        adj_bits does NOT include edge (g_u, g_v) yet.
        """
        # Mapped Q3 vertices to graph vertices
        mapping = [-1] * 8
        mapping[q3_u] = g_u
        mapping[q3_v] = g_v
        used_g = [False] * n
        used_g[g_u] = True
        used_g[g_v] = True
        remaining_q3 = [x for x in range(8) if x not in (q3_u, q3_v)]
        
        def backtrack(idx):
            if idx == len(remaining_q3):
                # All mapped. Verify all Q3 edges exist in G (including the new edge).
                for a, b in q3_edges:
                    ga = mapping[a]
                    gb = mapping[b]
                    if (ga == g_u and gb == g_v) or (ga == g_v and gb == g_u):
                        # This is the new edge, it's being added
                        continue
                    # Check if edge exists in adj_bits
                    if not (adj_bits[ga] & (1 << gb)):
                        return False
                return True
            
            q3_vertex = remaining_q3[idx]
            
            # Find candidates: graph vertices not yet used
            # and adjacent to all already-mapped Q3 neighbors of q3_vertex
            for g_cand in range(n):
                if used_g[g_cand]:
                    continue
                # Check adjacency to all mapped Q3 neighbors
                ok = True
                for q3_neighbor in q3_adj[q3_vertex]:
                    g_neighbor = mapping[q3_neighbor]
                    if g_neighbor != -1:
                        # If this is the new edge pair, skip the check
                        if (g_cand == g_u and g_neighbor == g_v) or (g_cand == g_v and g_neighbor == g_u):
                            continue
                        if not (adj_bits[g_cand] & (1 << g_neighbor)):
                            ok = False
                            break
                if ok:
                    mapping[q3_vertex] = g_cand
                    used_g[g_cand] = True
                    
                    if backtrack(idx + 1):
                        return True
                    
                    mapping[q3_vertex] = -1
                    used_g[g_cand] = False
            
            return False
        
        return backtrack(0)
    
    def local_search(adj_bits, max_iterations=500):
        """Try to add edges that don't create Q3, and remove edges that allow adding more."""
        current_count = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding each missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_bits[u] & (1 << v)):
                        if not creates_q3_fast(u, v, adj_bits):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            current_count += 1
                            improved = True
            
            if not improved:
                break
        
        return adj_bits, current_count
    
    def try_remove_and_add(adj_bits, current_count):
        """Try removing one edge and then greedily adding back to see if we get more."""
        best_adj = adj_bits[:]
        best_count = current_count
        
        # Get list of existing edges
        existing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_bits[u] & (1 << v):
                    existing_edges.append((u, v))
        
        random.shuffle(existing_edges)
        
        for u, v in existing_edges:
            if time.time() > deadline:
                break
            
            # Remove edge (u, v)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
            # Try to add back edges greedily
            trial_adj = adj_bits[:]
            trial_count = current_count - 1
            
            for u2 in range(n):
                for v2 in range(u2+1, n):
                    if not (trial_adj[u2] & (1 << v2)):
                        if not creates_q3_fast(u2, v2, trial_adj):
                            trial_adj[u2] |= (1 << v2)
                            trial_adj[v2] |= (1 << u2)
                            trial_count += 1
            
            if trial_count > best_count:
                best_count = trial_count
                best_adj = trial_adj[:]
            
            # Restore edge (u, v)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        return best_adj, best_count
    
    # Run multiple random trials
    trials = 30
    for t in range(trials):
        if time.time() > deadline:
            break
        
        random.seed(t * 7919 + 42)
        
        # Use bitmask adjacency for speed
        adj_bits = [0] * n
        
        # Start with K3,13
        for u in range(3):
            for v in range(3, 16):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Build list of remaining edges
        existing = set()
        for u in range(3):
            for v in range(3, 16):
                existing.add((u, v))
        
        remaining_edges = [e for e in all_edges if e not in existing]
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if not creates_q3_fast(u, v, adj_bits):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Local search: try to add more edges
        adj_bits, count = local_search(adj_bits)
        
        # Try remove-and-add local search
        adj_bits, count = try_remove_and_add(adj_bits, count)
        
        if count > best_count:
            best_count = count
            # Convert to boolean matrix for storage
            best_adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if adj_bits[i] & (1 << j):
                        best_adj[i][j] = True
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    
    return result
