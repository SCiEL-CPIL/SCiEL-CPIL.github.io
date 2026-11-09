import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    # Strategy:
    # 1. Construct a base graph that is "structurally" good for low book size.
    #    The Edwards-Khadzhiivanov-Nikiforov theorem suggests that graphs close to
    #    a complete bipartite graph or a complete graph minus a matching might be relevant,
    #    but for minimizing the MAX book, we want to avoid high common neighborhoods.
    #    
    #    A regular graph with low "triangle density" is better.
    #    Consider a 5-partite graph. Partition 40 vertices into 5 sets of 8.
    #    Connect all vertices in set i to all vertices in set i+1 mod 5.
    #    This is a blow-up of C5.
    #    Each vertex has degree 8+8=16.
    #    Total edges: 5 * 8 * 8 = 320 edges.
    #    We need 401 edges. We need to add 81 more edges.
    #    
    #    In the C5 blow-up, the book size for any edge (u,v) between part i and i+1
    #    is the number of common neighbors.
    #    u is in P_i, v is in P_{i+1}.
    #    Common neighbors must be in P_{i-1} (connected to u) AND P_{i+2} (connected to v)? No.
    #    Neighbors of u: P_{i-1} and P_{i+1}.
    #    Neighbors of v: P_i and P_{i+2}.
    #    Intersection: P_{i+1} \ {v} is not in Neighbors(u) unless u is connected to P_{i+1} (yes).
    #    Wait, u in P_i is connected to P_{i-1} and P_{i+1}.
    #    v in P_{i+1} is connected to P_i and P_{i+2}.
    #    Common neighbors of u and v:
    #    A vertex w is a common neighbor if w is neighbor of u AND w is neighbor of v.
    #    Neighbors of u: P_{i-1} U P_{i+1}.
    #    Neighbors of v: P_i U P_{i+2}.
    #    Intersection: (P_{i-1} U P_{i+1}) ∩ (P_i U P_{i+2}) = Empty set.
    #    So the C5 blow-up has 0 triangles! Book size 0 for all edges.
    #    
    #    This is excellent. We start with 320 edges, book size 0.
    #    We need to add 81 edges.
    #    Adding edges within a part or between non-adjacent parts creates triangles.
    #    
    #    Let's analyze adding an edge within a part P_i.
    #    If we add edge (u, v) with u, v in P_i.
    #    Neighbors of u: P_{i-1}, P_{i+1}.
    #    Neighbors of v: P_{i-1}, P_{i+1}.
    #    Common neighbors: P_{i-1} U P_{i+1}.
    #    Size: 8 + 8 = 16.
    #    So adding an intra-part edge creates a book of size 16. This is high.
    #    
    #    What if we add edges between non-adjacent parts? e.g. P_i and P_{i+2}.
    #    u in P_i, v in P_{i+2}.
    #    Neighbors of u: P_{i-1}, P_{i+1}.
    #    Neighbors of v: P_{i+1}, P_{i+3}.
    #    Common neighbors: P_{i+1}.
    #    Size: 8.
    #    So adding an edge between P_i and P_{i+2} creates a book of size 8.
    #    
    #    We need to add 81 edges.
    #    If we add all 81 edges between non-adjacent parts (distance 2 in C5),
    #    we can potentially keep book sizes lower than 16.
    #    However, adding multiple edges might increase common neighbors for other edges.
    #    
    #    Let's try to add edges between P_i and P_{i+2} systematically.
    #    There are 5 pairs of non-adjacent parts: (0,2), (1,3), (2,4), (3,0), (4,1).
    #    Each pair has 8*8=64 possible edges.
    #    Total 5*64 = 320 possible "extra" edges.
    #    We need 81.
    #    
    #    If we add edges only between non-adjacent parts, the graph remains "triangular" only
    #    in specific ways.
    #    Let's verify the book size for an original edge (u in P_i, v in P_{i+1}) after adding extras.
    #    Originally book size 0.
    #    Now, common neighbors of u and v might include vertices from other parts if edges are added.
    #    But we are only adding edges between P_i and P_{i+2} etc.
    #    Let's say we add edge (x in P_{i-1}, y in P_{i+1})? No, that's distance 2.
    #    Let's stick to adding edges between parts at distance 2.
    #    
    #    Construction:
    #    1. Base: C5 blow-up (320 edges).
    #    2. Add 81 edges from the set of distance-2 pairs.
    #    3. Use local search to minimize max book.
    
    # Define parts
    parts = [list(range(i*8, (i+1)*8)) for i in range(5)]
    
    adj = np.zeros((n, n), dtype=bool)
    present = set()
    
    # Add base C5 blow-up edges
    # Connect P_i to P_{i+1}
    for i in range(5):
        u_part = parts[i]
        v_part = parts[(i+1) % 5]
        for u in u_part:
            for v in v_part:
                adj[u, v] = True
                adj[v, u] = True
                present.add((min(u,v), max(u,v)))
    
    # We have 320 edges. Need 81 more.
    # Candidate edges: between P_i and P_{i+2}
    # Let's list them
    candidates = []
    for i in range(5):
        u_part = parts[i]
        v_part = parts[(i+2) % 5]
        for u in u_part:
            for v in v_part:
                candidates.append((min(u,v), max(u,v)))
                
    # Shuffle candidates to break symmetry
    np.random.shuffle(candidates)
    
    # Add 81 edges greedily? Or just take first 81?
    # Taking first 81 might create clusters.
    # Let's try to distribute them evenly among the 5 pairs of parts.
    # 81 / 5 = 16.2. So 16 or 17 per pair.
    
    # Actually, let's just add them and then optimize.
    # But we should be smart about WHICH edges to add initially.
    # Adding edges that don't share endpoints might be better?
    # With 81 edges and 40 vertices, it's hard to avoid sharing.
    
    # Let's just add 81 random candidates from the distance-2 set.
    for edge in candidates[:81]:
        u, v = edge
        adj[u, v] = True
        adj[v, u] = True
        present.add(edge)
        
    # Now we have 401 edges.
    # Current max book size?
    # For an intra-part edge (if any added, we didn't add intra-part), book is high.
    # We only added distance-2 edges.
    # Let's compute max book.
    
    def max_book(adj_mat):
        max_b = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v]:
                    b = int(np.sum(adj_mat[u] & adj_mat[v]))
                    if b > max_b:
                        max_b = b
        return max_b
        
    current_score = max_book(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    # Local Search
    # We want to reduce max book.
    # Move: Remove an edge (u,v) and add an edge (x,y).
    # Prefer moves that reduce the book size of the edge with max book.
    
    # To speed up, we can track the book sizes of all edges?
    # n=40, edges=401.
    # Recomputing max_book is O(E * N) ~ 401 * 40 = 16000 ops.
    # 110s / 16000 ~ 6800 steps. That's small.
    # We need a faster max_book or a smarter search.
    
    # Faster max_book using numpy:
    # Triangles = sum of (A^3) / 6? No, we need book per edge.
    # Book size of (u,v) is (A[u] & A[v]).sum()
    
    # Let's try a heuristic local search.
    # Identify the edge with the maximum book.
    # Try to "break" its triangles by removing one of its incident edges?
    # No, we must keep edge count constant.
    # So we must swap.
    
    # If edge (u,v) has book size B, it means there are B vertices w such that (u,w) and (v,w) exist.
    # To reduce book size of (u,v), we can remove an edge (u,w) or (v,w).
    # If we remove (u,w), we must add some other edge (x,y).
    # We should choose (x,y) such that it doesn't create large books.
    # Ideally, (x,y) should have small book size.
    
    # Strategy:
    # 1. Find edge (u,v) with max book.
    # 2. Pick a common neighbor w of u and v.
    # 3. Consider removing (u,w) or (v,w).
    # 4. Consider adding a random non-edge (x,y) that has low current book size.
    # 5. Accept if max book decreases.
    
    T = 1.0
    alpha = 0.99
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Find max book edge
        max_b = 0
        max_edge = None
        # We can speed this up by only checking edges that might have changed,
        # but for simplicity, we scan all.
        # Optimization: Maintain a list of edges.
        edges_list = list(present)
        
        # To avoid O(E*N) per step, we can just compute max book for the current graph.
        # Since N=40, it's fast enough for ~10k steps?
        # 10k * 16000 = 160M ops. Might be too slow in Python loops.
        # Let's use numpy vectorization for max_book.
        
        # A[u] & A[v] is a boolean vector.
        # We can compute all pair common neighbors at once?
        # C = A @ A.T gives number of common neighbors for ALL pairs (including non-edges).
        # Book size for edge (u,v) is C[u,v].
        # We only care about edges.
        
        C = adj @ adj.T
        # C is 40x40.
        # Mask out non-edges and diagonal.
        # Create a mask of edges.
        edge_mask = adj.copy()
        np.fill_diagonal(edge_mask, False)
        
        # Get book sizes for all edges
        # We can just take the max of C where edge_mask is True.
        # But we need to be careful: C[u,v] counts common neighbors.
        # This is exactly the book size.
        
        current_max = int(np.max(C[edge_mask]))
        
        if current_max < best_score:
            best_score = current_max
            best_adj = adj.copy()
            if best_score == 7: # Theoretical min
                break
            
        # If we hit the lower bound, stop.
        # Lower bound is 7.
        
        # Select an edge to remove: One that is part of many triangles?
        # Or just the edge with max book?
        # If we remove (u,v) with max book, we break all its triangles.
        # But we must add an edge.
        
        # Let's pick the edge with the highest book size.
        # Flatten C[edge_mask] to find indices.
        # This is expensive.
        # Alternative: Sample a few edges and pick the one with high book.
        
        # Let's pick a random edge from the current set and check its book size.
        # Or, better, pick the edge with max book from C.
        
        # Find indices of max in C[edge_mask]
        # C_masked = np.where(edge_mask, C, -1)
        # max_val = np.max(C_masked)
        # idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
        # u_rem, v_rem = idx[0], idx[1]
        
        # This is O(N^2) = 1600 ops. Very fast.
        
        C_masked = np.where(edge_mask, C, -1)
        max_val = int(np.max(C_masked))
        
        if max_val == 0:
            break # No triangles, perfect (but impossible with 401 edges? Maybe)
            
        idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
        u_rem, v_rem = idx[0], idx[1]
        
        # We want to remove (u_rem, v_rem).
        # But maybe removing it doesn't help if the new edge creates a bigger book.
        # Let's try to add an edge that has low book size.
        
        # Find a non-edge with low book size.
        # Non-edges are where adj is False and u!=v.
        # Book size of a potential edge (x,y) is C[x,y].
        # We want to add (x,y) with small C[x,y].
        
        # Mask non-edges
        non_edge_mask = ~edge_mask
        np.fill_diagonal(non_edge_mask, False)
        
        # We want to minimize C[x,y] for x,y in non_edge_mask.
        # But we also want the move to reduce the global max.
        # Removing (u_rem, v_rem) reduces book sizes of its neighbors?
        # Actually, removing an edge (u,v) reduces the book size of (u,v) to 0 (it's gone).
        # It also reduces the book size of other edges that shared triangles with (u,v)?
        # No, book size of (u,w) depends on common neighbors of u and w.
        # If v was a common neighbor of u and w, then (u,v) and (v,w) existed.
        # Removing (u,v) removes v from the common neighbors of u and w?
        # No, v is a vertex. The edge (u,v) is removed.
        # The common neighbors of u and w are vertices z such that (u,z) and (w,z) exist.
        # Removing (u,v) does not change whether (w,v) exists.
        # It only changes the set of edges incident to u and v.
        
        # So removing (u_rem, v_rem) only affects the book sizes of edges incident to u_rem or v_rem.
        
        # Let's just try a random non-edge with low C value.
        
        # Get list of non-edges
        non_edges = np.argwhere(non_edge_mask)
        if len(non_edges) == 0:
            break
            
        # Sample a few non-edges
        sample_size = min(10, len(non_edges))
        sample_idx = np.random.choice(len(non_edges), sample_size, replace=False)
        sampled_non_edges = non_edges[sample_idx]
        
        # Pick the one with smallest C value
        best_add = None
        min_c = float('inf')
        for x, y in sampled_non_edges:
            c_val = C[x, y]
            if c_val < min_c:
                min_c = c_val
                best_add = (x, y)
                
        if best_add is None:
            continue
            
        u_add, v_add = best_add
        
        # Apply move
        adj[u_rem, v_rem] = False
        adj[v_rem, u_rem] = False
        present.discard((min(u_rem, v_rem), max(u_rem, v_rem)))
        
        adj[u_add, v_add] = True
        adj[v_add, u_add] = True
        present.add((min(u_add, v_add), max(u_add, v_add)))
        
        # Check new score
        # We can just let the next loop iteration compute it.
        # But we might want to accept/reject.
        # Since we are greedily removing the max book edge, it should help.
        # However, adding an edge might create a new max.
        
        # Simple hill climbing: keep the move.
        # If it makes things worse, we might oscillate.
        # To be safe, we can compute new max and revert if it's worse.
        
        C_new = adj @ adj.T
        edge_mask_new = adj.copy()
        np.fill_diagonal(edge_mask_new, False)
        
        new_max = int(np.max(C_new[edge_mask_new]))
        
        if new_max > current_max:
            # Revert
            adj[u_rem, v_rem] = True
            adj[v_rem, u_rem] = True
            present.add((min(u_rem, v_rem), max(u_rem, v_rem)))
            
            adj[u_add, v_add] = False
            adj[v_add, u_add] = False
            present.discard((min(u_add, v_add), max(u_add, v_add)))
            
            # If we can't improve, maybe try a different removal?
            # For now, just break or continue.
            # Let's continue to try other moves.
            pass
        else:
            # Keep move
            # Update best if improved
            if new_max < best_score:
                best_score = new_max
                best_adj = adj.copy()
                if best_score == 7:
                    break

    # Return best_adj
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                result.append([u, v])
                
    return result
