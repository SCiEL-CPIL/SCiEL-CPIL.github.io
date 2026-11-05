import time
import sys

def construct():
    # Step 1: Construct the Grötzsch graph on vertices 0..10
    # This graph is triangle-free and not 3-colorable.
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Mycielski construction
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the hub
    for i in range(5):
        u = i
        w = (i + 1) % 5
        # Edges between original and copies
        # In Grötzsch, vertex i is connected to copy of i-1 and i+1?
        # Standard Mycielski: edge (u,v) in G implies edges (u, v') and (v, u') in mu(G).
        # Cycle edges: (0,1), (1,2), (2,3), (3,4), (4,0)
        # (0,1) -> (0, 1') and (1, 0') => (0,6) and (1,5)
        # (1,2) -> (1, 2') and (2, 1') => (1,7) and (2,6)
        # (2,3) -> (2, 3') and (3, 2') => (2,8) and (3,7)
        # (3,4) -> (3, 4') and (4, 3') => (3,9) and (4,8)
        # (4,0) -> (4, 0') and (0, 4') => (4,5) and (0,9)
        
        # The seed code did:
        # for u, w in cycle:
        #    edges += [[u, 5 + w], [w, 5 + u]]
        # This matches the logic above.
        edges.append([u, 5 + w])
        edges.append([w, 5 + u])
        
    # Hub 10 connected to all copies 5..9
    for i in range(5):
        edges.append([5 + i, 10])
        
    # Total edges so far: 5 (cycle) + 10 (spokes) + 5 (hub) = 20.
    
    # Step 2: Build adjacency bitmasks for the core (0..10)
    n_core = 11
    adj = [0] * n_core
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Step 3: Precompute independent sets
    # A set S is independent if for all u in S, (adj[u] & S) == 0?
    # More efficiently: S is independent if for all u in S, (adj[u] & S) == 0 is not quite right because adj[u] contains u? No, simple graph.
    # Check: for all u, v in S, u != v, (1 << v) not in adj[u].
    
    def is_independent(mask):
        # mask is an integer representing the set of vertices
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Check if any other vertex in mask is adjacent to v
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Find all independent sets
    independent_sets = []
    for mask in range(1 << n_core):
        if is_independent(mask):
            independent_sets.append(mask)
            
    # Step 4: Find best pair of disjoint independent sets A, B and split a
    # We have 5 new vertices (11..15).
    # We split them into two groups of size a and 5-a.
    # Group 1 connects to A, Group 2 connects to B.
    # Edges added: a*|A| + (5-a)*|B| + a*(5-a)
    # We want to maximize this.
    
    best_score = -1
    best_A = 0
    best_B = 0
    best_a = 0
    
    # Iterate over all pairs of disjoint independent sets
    # To optimize, we can iterate A, then B from independent_sets where (A & B) == 0
    for A in independent_sets:
        # Calculate size of A
        size_A = bin(A).count("1")
        for B in independent_sets:
            if A & B:
                continue
            size_B = bin(B).count("1")
            # Try all splits a from 0 to 5
            for a in range(6):
                b = 5 - a
                # Edges between new vertices and core
                e_core = a * size_A + b * size_B
                # Edges between new vertices
                e_new = a * b
                total_new = e_core + e_new
                
                if total_new > best_score:
                    best_score = total_new
                    best_A = A
                    best_B = B
                    best_a = a
                    
    # Step 5: Construct the final graph
    final_edges = edges.copy()
    
    new_vertices = [11, 12, 13, 14, 15]
    
    # Group 1: first best_a vertices
    # Group 2: remaining 5 - best_a vertices
    
    # Add edges from Group 1 to A
    for i in range(best_a):
        v_new = new_vertices[i]
        # Iterate bits in best_A
        mask = best_A
        while mask:
            lsb = mask & -mask
            v_core = lsb.bit_length() - 1
            final_edges.append([v_new, v_core])
            mask ^= lsb
            
    # Add edges from Group 2 to B
    for i in range(best_a, 5):
        v_new = new_vertices[i]
        mask = best_B
        while mask:
            lsb = mask & -mask
            v_core = lsb.bit_length() - 1
            final_edges.append([v_new, v_core])
            mask ^= lsb
            
    # Add edges between Group 1 and Group 2
    for i in range(best_a):
        for j in range(best_a, 5):
            final_edges.append([new_vertices[i], new_vertices[j]])
            
    # Step 6: Verification (Optional but recommended for safety in production-like code)
    # Check triangle-free
    n_total = 16
    adj_total = [0] * n_total
    for u, v in final_edges:
        adj_total[u] |= (1 << v)
        adj_total[v] |= (1 << u)
        
    # Triangle check: for each edge (u,v), check if any w is neighbor of both
    # This is O(E * N) or similar. With ~50 edges and 16 nodes, it's fast.
    is_tf = True
    for u in range(n_total):
        for v in range(u+1, n_total):
            if adj_total[u] & (1 << v):
                # Check common neighbors
                common = adj_total[u] & adj_total[v]
                if common:
                    is_tf = False
                    break
        if not is_tf:
            break
            
    if not is_tf:
        # Fallback to seed if logic fails (should not happen)
        return edges

    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
