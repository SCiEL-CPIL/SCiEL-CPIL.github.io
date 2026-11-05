import time
import random
import numpy as np

_DEADLINE = time.time() + 130.0

def _has_c5(n, adj):
    """Check if graph contains a C5 using bitset operations."""
    # adj is a list of bitmasks, one per vertex
    # For each edge (u,v), check if there's a path u-a-b-c-v with all distinct
    for u in range(n):
        # Get neighbors of u as bitmask
        nu = adj[u]
        # For each neighbor a of u (a != v, we'll handle v in the loop)
        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            if a == u:
                continue
            # For each neighbor b of a (b != u, b != v)
            na = adj[a] & ~a_bit & ~(1 << u)
            while na:
                b_bit = na & (-na)
                b = b_bit.bit_length() - 1
                na ^= b_bit
                if b == a:
                    continue
                # For each neighbor c of b (c != a, c != u)
                nb = adj[b] & ~b_bit & ~(1 << a) & ~(1 << u)
                while nb:
                    c_bit = nb & (-nb)
                    c = c_bit.bit_length() - 1
                    nb ^= c_bit
                    if c == b:
                        continue
                    # Check if c is adjacent to u (closing the cycle)
                    if adj[c] & (1 << u):
                        # Verify all 5 vertices distinct: u, a, b, c, and the edge c-u
                        # u, a, b, c are distinct by construction
                        # Need to check that the path doesn't create a shorter cycle issue
                        # Actually for C5 as subgraph, we just need 5 distinct vertices in a cycle
                        # u-a-b-c-u is a 4-cycle, not 5. We need u-a-b-c-v-u where v is another vertex
                        # Wait, I made an error. Let me reconsider.
                        pass
    return False

def _has_c5_correct(n, adj):
    """
    Check if graph contains a C5 (cycle of length 5).
    adj is a list of bitmasks.
    A C5 is 5 distinct vertices v0,v1,v2,v3,v4 with edges v0v1,v1v2,v2v3,v3v4,v4v0.
    """
    for v0 in range(n):
        # v1 must be neighbor of v0
        v1_mask = adj[v0] & ~(1 << v0)
        while v1_mask:
            v1_bit = v1_mask & (-v1_mask)
            v1 = v1_bit.bit_length() - 1
            v1_mask ^= v1_bit
            # v2 must be neighbor of v1, not v0, not v1
            v2_mask = adj[v1] & ~(1 << v0) & ~(1 << v1)
            while v2_mask:
                v2_bit = v2_mask & (-v2_mask)
                v2 = v2_bit.bit_length() - 1
                v2_mask ^= v2_bit
                # v3 must be neighbor of v2, not v0,v1,v2
                v3_mask = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                while v3_mask:
                    v3_bit = v3_mask & (-v3_mask)
                    v3 = v3_bit.bit_length() - 1
                    v3_mask ^= v3_bit
                    # v4 must be neighbor of v3, not v0,v1,v2,v3
                    # and v4 must be neighbor of v0
                    v4_mask = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                    v4_mask &= adj[v0]  # v4 must connect to v0
                    if v4_mask:
                        return True
    return False

def _has_independent_set_9(n, adj):
    """
    Check if graph has an independent set of size 9.
    adj is a list of bitmasks where adj[i] has bits for neighbors of i.
    Independent set: no two vertices are adjacent.
    """
    # Complement: non-neighbors (excluding self)
    all_mask = (1 << n) - 1
    comp = []
    for i in range(n):
        # Non-neighbors of i, excluding i itself
        comp.append(all_mask & ~(adj[i] | (1 << i)))
    
    # Backtracking search for independent set of size 9
    # Use ordering by degree in complement (fewest non-neighbors first? or most?)
    # For finding independent set, we want to pick vertices that have few constraints
    # Actually, for the complement graph (where edges = non-edges in G), we want a clique of size 9
    # Let's use a simple backtracking with pruning
    
    # Order vertices: try vertices with high degree in complement first (more non-neighbors)
    # Actually for clique finding, high degree first is often good
    order = sorted(range(n), key=lambda v: comp[v].bit_count(), reverse=True)
    
    def search(start_idx, current_set, current_size):
        if current_size == 9:
            return True
        if start_idx >= n:
            return False
        # Pruning: not enough vertices left
        if n - start_idx < 9 - current_size:
            return False
        
        for i in range(start_idx, n):
            v = order[i]
            # Check if v is compatible with all vertices in current_set
            compatible = True
            for u in current_set:
                if adj[u] & (1 << v):  # u and v are adjacent in G, so can't both be in independent set
                    compatible = False
                    break
            if not compatible:
                continue
            # Try including v
            if search(i + 1, current_set + [v], current_size + 1):
                return True
        return False
    
    return search(0, [], 0)

def _validate(n, adj_bitmasks):
    """Validate that graph is C5-free and has no independent set of size 9."""
    if n < 1 or n > 35:
        return False
    if _has_c5_correct(n, adj_bitmasks):
        return False
    if _has_independent_set_9(n, adj_bitmasks):
        return False
    return True

def _bitmasks_to_edges(n, adj_bitmasks):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    
    # Start with the known 32-vertex solution: 8 disjoint K4s
    best_n = 32
    best_adj = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            mask = 0
            for j in range(4):
                if i != j:
                    mask |= (1 << (base + j))
            best_adj.append(mask)
    
    # Verify the 32-vertex solution is valid
    if not _validate(32, best_adj):
        # Should not happen, but fallback
        pass
    
    # Try to find n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 15:
            break
        
        found = False
        
        # Strategy 1: Start with 8 K4s + new vertex, try to add edges to new vertex
        # The new vertex must not create a C5 and must reduce alpha from 9 to <=8
        # In 8 K4s, alpha = 8. Adding one isolated vertex makes alpha = 9.
        # We need to connect the new vertex to at least one vertex in each K4
        # (or enough vertices to prevent picking one from each K4 + the new vertex)
        
        # Actually, if the new vertex is connected to at least one vertex in each of the 8 K4s,
        # then any independent set containing the new vertex can contain at most 7 other vertices
        # (one from each K4 except the one where the new vertex is connected).
        # Wait, that's not quite right. Let me think more carefully.
        
        # If new vertex v is connected to vertex x_b in K4_b for each block b,
        # then an independent set containing v can contain at most one vertex from each K4_b,
        # but NOT x_b. So it can contain at most 3 from K4_b (the other 3 vertices).
        # But we can still pick at most 1 from each K4_b (since K4 is a clique).
        # So max independent set containing v is 1 + 7 = 8 (pick v + one from 7 K4s, avoiding the connected one).
        # Actually we can pick one from each K4 except we must avoid x_b in each.
        # Since K4 is a clique, we can pick at most 1 from each K4.
        # If v is connected to x_b in K4_b, we can still pick one of the other 3 from K4_b.
        # So we can pick v + one from each of 8 K4s = 9 vertices? No, because v is connected to x_b,
        # but we're picking a different vertex from K4_b, so that's fine.
        # Wait, if we pick v and a vertex y_b from K4_b where y_b != x_b, that's fine since v is only connected to x_b.
        # So we can have v + y_1 + y_2 + ... + y_8 = 9 vertices, all independent!
        # So connecting to one vertex per K4 is NOT enough.
        
        # We need to connect v to ALL vertices in at least one K4, or use a different strategy.
        # If v is connected to all 4 vertices of K4_b, then any independent set containing v
        # cannot contain any vertex from K4_b. So max is 1 + 7 = 8.
        
        # Strategy: Connect new vertex to all 4 vertices of one K4.
        # But this might create a C5. Let's check.
        # If v is connected to all of K4_b = {a,b,c,d}, then we have a K5 on {v,a,b,c,d}.
        # K5 contains C5! So this creates a C5.
        
        # Alternative: Use a different base structure.
        # Try: 7 K4s + one K5? K5 has C5. No.
        # Try: 7 K4s + one component of 5 vertices that is C5-free and has alpha <= 1?
        # A K5 has alpha 1 but has C5. A K4 + isolated has alpha 2.
        # What about a 5-vertex C5-free graph with alpha <= 2?
        # K4 + isolated: alpha = 1 + 1 = 2. C5-free. Total: 7*4 + 5 = 33.
        # Alpha = 7*1 + 2 = 9. Still too high!
        
        # What about 7 K4s + one 5-vertex graph with alpha = 1? That's K5, which has C5.
        # 7 K4s + one 5-vertex graph with alpha = 0? Impossible for n>=1.
        
        # So 8 K4s + 1 vertex doesn't work easily.
        
        # Strategy 2: Use a different construction entirely.
        # Try: Complete multipartite graphs? No, those have C5 for certain part sizes.
        
        # Strategy 3: Randomized search starting from 8 K4s, add vertex and edges.
        # Use simulated annealing or local search.
        
        # Let's try a different approach: start from scratch with random graphs and local search.
        # Or try specific known constructions.
        
        # Actually, let's try: Can we modify the 8 K4s structure?
        # What if we merge two K4s into a larger C5-free component?
        # Two K4s = 8 vertices. Can we make a C5-free graph on 8 vertices with alpha <= 2?
        # K4,4 has C5 (actually K4,4 has C4 but does it have C5? K4,4 is bipartite, so no odd cycles, no C5).
        # K4,4 alpha = 4. Too high.
        # What about K4 + K4 connected by some edges?
        
        # Let's try a more systematic approach: local search on the edge set.
        
        # For now, let's try a few specific constructions for n=33:
        # Construction: 8 K4s, but replace one K4 with a 5-vertex C5-free graph with alpha=1?
        # That requires K5, which has C5.
        
        # Try: 7 K4s + 1 K4 + 1 vertex connected to all of one K4?
        # That's 33 vertices. But as shown, this may create C5 or not reduce alpha enough.
        
        # Let's just do a randomized search with the bitset-based validators.
        
        # Start with 8 K4s + isolated vertex, then try to add edges to the isolated vertex.
        adj = list(best_adj)  # 32 vertices
        # Add vertex 32 (isolated)
        adj.append(0)
        n = target_n
        
        # We need to add edges to vertex 32 to reduce alpha from 9 to <=8.
        # And avoid creating C5.
        
        # Try adding edges from vertex 32 to various vertices.
        # We need to hit enough K4s to prevent alpha=9.
        
        # Let's try: connect vertex 32 to all 4 vertices of K4_0.
        # This creates K5 on {32, 0,1,2,3}, which has C5. Bad.
        
        # Connect vertex 32 to 3 vertices of K4_0 and 1 vertex of K4_1?
        # Let's just do a search.
        
        # Randomized local search
        for attempt in range(200):
            if time.time() > _DEADLINE - 15:
                break
            
            # Start with 8 K4s + isolated vertex 32
            adj = list(best_adj)
            adj.append(0)
            
            # Randomly add edges from vertex 32 to other vertices
            # We want to add enough edges to reduce alpha, but not create C5.
            
            # Strategy: connect to a random subset of vertices
            # Try different numbers of edges
            num_edges_to_add = random.randint(4, 12)
            candidates = list(range(32))
            random.shuffle(candidates)
            to_connect = candidates[:num_edges_to_add]
            
            for v in to_connect:
                adj[32] |= (1 << v)
                adj[v] |= (1 << 32)
            
            # Check if valid
            if _validate(33, adj):
                found = True
                best_n = 33
                best_adj = adj
                break
            
            # If not valid, try a different random subset
        
        if found:
            continue  # Try n=34
        
        # If n=33 not found, try n=34 and 35 with similar approach
        # But given time constraints, just return what we have
    
    # Convert best_adj to edges
    edges = _bitmasks_to_edges(best_n, best_adj)
    
    return {'n': best_n, 'edges': edges}
