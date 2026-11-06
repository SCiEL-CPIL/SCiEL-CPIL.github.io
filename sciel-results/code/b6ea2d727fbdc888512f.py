import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 320  # Safety margin
    
    # Target: 42 vertices (Exoo's graph)
    # If we can't find 42, we try to find the largest valid n.
    # But the search is for a specific n. 
    # Strategy: Try to find a (5,5)-free graph on 42 vertices.
    # If successful, return 42.
    # If not, fall back to seed (37).
    
    n_target = 42
    seed_q = 37
    
    # Initialize with Paley graph padded to n_target
    # Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # Build adjacency matrix for n_target
    # We will use a list of lists of bools for simplicity, 
    # but for speed in the inner loop, we might want bitmasks.
    # Let's use bitmasks (integers).
    
    adj = [0] * n_target
    
    # Initialize with Paley structure for first 37 vertices
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initialize remaining vertices (37 to 41) randomly
    # Or set them to 0 (isolated) and let the search connect them?
    # Random is better for connectivity.
    for i in range(q, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            # Also connect to previous vertices?
            # The random initialization above only handled i,j >= q.
            # We need to connect new vertices to old ones too.
            # Let's re-initialize the whole graph randomly? 
            # No, starting from Paley is a good prior.
            # But we need edges between [0..36] and [37..41].
            # Let's just randomize the whole graph? 
            # Exoo's graph is specific. 
            # Let's try random start with p=0.5. It's more robust.
            pass
            
    # Let's restart with a pure random graph for simplicity and robustness of the search
    # The Paley structure might be a local trap.
    adj = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Precompute pairs and triples for fast delta calculation
    # For each pair (u, v), we need to check all triples (w, x, y) from the remaining n-2 vertices.
    # We can precompute the list of triples for each pair? 
    # That's 861 pairs * 9880 triples = 8.5M triples stored. 
    # Memory is fine, but generation time might be high.
    # Let's generate on the fly or use a generic list of all triples?
    # No, the triple must be disjoint from u,v.
    
    # Optimization: 
    # Instead of precomputing, we iterate w, x, y from the set of all vertices excluding u, v.
    # We can precompute a list of all vertices `all_v = list(range(n_target))`.
    # For a given u, v, the candidates are `all_v` minus u and v.
    # We can use itertools.combinations on the filtered list.
    
    all_vertices = list(range(n_target))
    
    def count_bad_sets(adj):
        """Count K5 and independent 5-sets."""
        bad_count = 0
        for combo in itertools.combinations(range(n_target), 5):
            u, v, w, x, y = combo
            # Check edges
            # Using bitmasks is faster
            # But we need to check 10 edges.
            # Let's just sum them.
            e = 0
            e += 1 if (adj[u] >> v) & 1 else 0
            e += 1 if (adj[u] >> w) & 1 else 0
            e += 1 if (adj[u] >> x) & 1 else 0
            e += 1 if (adj[u] >> y) & 1 else 0
            e += 1 if (adj[v] >> w) & 1 else 0
            e += 1 if (adj[v] >> x) & 1 else 0
            e += 1 if (adj[v] >> y) & 1 else 0
            e += 1 if (adj[w] >> x) & 1 else 0
            e += 1 if (adj[w] >> y) & 1 else 0
            e += 1 if (adj[x] >> y) & 1 else 0
            
            if e == 0 or e == 10:
                bad_count += 1
        return bad_count

    # Initial bad count
    current_bad = count_bad_sets(adj)
    
    # If we start with 0 bad sets, we are done. (Unlikely)
    
    # Local Search
    # We want to minimize current_bad.
    
    # Precompute candidates for each pair to avoid itertools overhead in the hot loop?
    # Let's try to generate the list of (w,x,y) for each (u,v) once.
    # This takes time but speeds up the loop.
    
    pair_triples = {}
    for u in range(n_target):
        for v in range(u + 1, n_target):
            others = [x for x in all_vertices if x != u and x != v]
            # We need combinations of 3 from 'others'
            # There are 40 others. C(40,3) = 9880.
            # Storing 9880 tuples * 861 pairs is ~8.5M tuples.
            # This might be too slow to build (seconds) and memory heavy.
            # Let's generate on the fly but optimize the inner loop.
            pass
            
    # Alternative: Just iterate all 5-sets in the delta calc?
    # No, that's C(42,5) = 850k checks. Too slow.
    # The pair-based delta is C(40,3) = 9880 checks. Much better.
    
    # Let's optimize the pair-based delta.
    # We need a fast way to get the 3-combinations of the remaining 40 vertices.
    
    # We can precompute the list of indices for each vertex to be excluded?
    
    # Let's just use a simple loop structure.
    
    temperature = 0.1
    cooling = 0.999
    
    best_adj = adj[:]
    best_bad = current_bad
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Check if edge exists
        edge_exists = (adj[u] >> v) & 1
        
        # Calculate delta
        # We need to check all triples (w, x, y) from V \ {u, v}
        # To do this efficiently, we can iterate w, x, y.
        # Since n=42, we can just use nested loops or itertools.
        
        delta = 0
        
        # Create a list of valid vertices for the triple
        # This list creation is expensive. 
        # Let's try to hardcode the iteration or use a precomputed mask?
        
        # A faster way:
        # The number of edges in the 5-set {u,v,w,x,y} is:
        # e_uv + e_uw + e_ux + e_uy + e_vw + e_vx + e_vy + e_wx + e_wy + e_xy
        
        # We can precompute for each pair (u,v) the "contribution" of the other edges?
        # No, it depends on w,x,y.
        
        # Let's just run the loop. 9880 iterations.
        # We need to iterate over all combinations of 3 from the 40 other vertices.
        
        # To avoid generating the list of others every time, we can precompute 
        # the list of all vertices, and skip u and v in the loop?
        # Or precompute `others_for_pair`?
        
        # Let's precompute `others_for_pair` as a list of lists of ints.
        # It's a one-time cost.
        
        # We'll do this outside the main loop if not already done.
        # But we defined pair_triples as empty above. Let's populate it.
        
        if not pair_triples:
             for u_idx in range(n_target):
                 for v_idx in range(u_idx + 1, n_target):
                     others = [x for x in all_vertices if x != u_idx and x != v_idx]
                     # Generate combinations
                     # This is slow in Python.
                     # But we only do it once.
                     # C(40,3) = 9880.
                     # 861 * 9880 = 8.5M tuples.
                     # Might take 10-20s to build. Acceptable.
                     triples = list(itertools.combinations(others, 3))
                     pair_triples[(u_idx, v_idx)] = triples
                     # Symmetric
                     pair_triples[(v_idx, u_idx)] = triples # Same list object? No, distinct content but same values.
                     # Actually, the set of others is the same. So we can reuse the list object?
                     # Yes, the list of triples is identical for (u,v) and (v,u).
        
        # Get the list of triples for this pair
        triples = pair_triples.get((u, v))
        if triples is None:
             # Should not happen
             continue
             
        for w, x, y in triples:
            # Calculate edge count for {u,v,w,x,y}
            # Current state
            e = 1 if edge_exists else 0
            e += 1 if (adj[u] >> w) & 1 else 0
            e += 1 if (adj[u] >> x) & 1 else 0
            e += 1 if (adj[u] >> y) & 1 else 0
            e += 1 if (adj[v] >> w) & 1 else 0
            e += 1 if (adj[v] >> x) & 1 else 0
            e += 1 if (adj[v] >> y) & 1 else 0
            e += 1 if (adj[w] >> x) & 1 else 0
            e += 1 if (adj[w] >> y) & 1 else 0
            e += 1 if (adj[x] >> y) & 1 else 0
            
            old_bad = 1 if (e == 0 or e == 10) else 0
            
            # New state: flip edge (u,v)
            new_e = e + (1 if not edge_exists else -1)
            new_bad = 1 if (new_e == 0 or new_e == 10) else 0
            
            delta += new_bad - old_bad
            
        # Acceptance
        if delta < 0:
            # Always accept improvement
            # Flip edge
            if edge_exists:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            current_bad += delta
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < pow(2.718, -delta / temperature):
                if edge_exists:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                current_bad += delta
        
        if current_bad < best_bad:
            best_bad = current_bad
            best_adj = adj[:]
            
        if best_bad == 0:
            break
            
        # Cooling
        temperature *= cooling
        
    # Convert best_adj to edge list
    if best_bad == 0:
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if (best_adj[i] >> j) & 1:
                    edges.append((i, j))
        return n_target, edges
    else:
        # Fallback to seed
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, seed_edges
