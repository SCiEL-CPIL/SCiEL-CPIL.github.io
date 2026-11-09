import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    n = 20
    # M[i] is a bitmask of neighbors of i
    M = [0] * n
    
    # Precompute pairs of vertices for faster iteration
    # pairs is a list of (u, v) with u < v
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding edge (u, v) is valid
    # We maintain the graph in M.
    # A K3,3 subgraph exists if there are two sets of 3 vertices {a,b,c} and {d,e,f}
    # such that all 9 edges between them exist.
    # Equivalently, no three vertices have 3 common neighbors.
    
    def can_add(u, v):
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False
        
        # If we add edge (u, v), the neighborhood of u gains v, and v gains u.
        # We need to check if this creates a K3,3.
        # A K3,3 containing the new edge (u, v) must have u and v in opposite parts.
        # Let the parts be A and B.
        # Case 1: u in A, v in B.
        # Then we need 2 more vertices in A (call them a1, a2) and 2 more in B (b1, b2).
        # Edges required:
        # u-a1, u-a2 (u is connected to a1, a2)
        # v-b1, v-b2 (v is connected to b1, b2)
        # a1-b1, a1-b2, a2-b1, a2-b2 (complete bipartite between {a1,a2} and {b1,b2})
        # Also a1-v? No, a1 is in A, v in B. So a1-v must exist?
        # Wait, K3,3 is a complete bipartite graph K_{3,3}.
        # If u in A, v in B, then:
        # A = {u, a1, a2}, B = {v, b1, b2}
        # Edges in A-B:
        # u-v (new)
        # u-b1, u-b2 (u connected to b1, b2)
        # a1-v, a2-v (a1, a2 connected to v)
        # a1-b1, a1-b2, a2-b1, a2-b2
        
        # So the conditions are:
        # 1. u is connected to b1, b2.
        # 2. v is connected to a1, a2.
        # 3. a1, a2 are connected to b1, b2.
        # 4. a1, a2 are NOT connected to each other (they are in same part A).
        # 5. b1, b2 are NOT connected to each other (they are in same part B).
        
        # Let's iterate over potential b1, b2 (neighbors of u, excluding v)
        # and a1, a2 (neighbors of v, excluding u)
        
        # Neighbors of u (excluding v): N_u = M[u] & ~(1 << v)
        # Neighbors of v (excluding u): N_v = M[v] & ~(1 << u)
        
        N_u = M[u] & ~(1 << v)
        N_v = M[v] & ~(1 << u)
        
        # Convert to lists of vertices for iteration
        # For efficiency, we can iterate over bits.
        
        # For each pair b1, b2 from N_u:
        #   For each pair a1, a2 from N_v:
        #     Check if a1, a2 are connected to b1, b2.
        #     Check if a1, a2 are not connected to each other.
        #     Check if b1, b2 are not connected to each other.
        
        # This is O(deg(u)^2 * deg(v)^2). Max deg is 19. 19^4 ~ 130k ops.
        # With 190 possible edges to add, and local search steps, this might be slow in Python.
        # We can optimize using bitmasks.
        
        # For a fixed pair b1, b2 from N_u:
        # We need a1, a2 in N_v such that:
        # a1 is connected to b1 and b2.
        # a2 is connected to b1 and b2.
        # a1 not connected to a2.
        # b1 not connected to b2.
        
        # Let's precompute common neighbors or use bit operations.
        
        # Iterate b1 in N_u
        b1_mask = N_u
        while b1_mask:
            b1 = (b1_mask & -b1_mask).bit_length() - 1
            b1_mask &= b1_mask - 1
            
            # b2 in N_u, b2 > b1 to avoid duplicates? No, just distinct.
            # But b1, b2 are in B, so they should not be connected.
            if (M[b1] >> b2) & 1: # placeholder, need to check b2
                pass
            
            b2_mask = N_u & ~(1 << b1)
            while b2_mask:
                b2 = (b2_mask & -b2_mask).bit_length() - 1
                b2_mask &= b2_mask - 1
                
                # Check if b1 and b2 are connected. If yes, they can't be in same part.
                if (M[b1] >> b2) & 1:
                    continue
                
                # Now we need a1, a2 in N_v such that:
                # a1 connected to b1 and b2.
                # a2 connected to b1 and b2.
                # a1 not connected to a2.
                
                # Candidates for a1, a2 are in N_v & M[b1] & M[b2]
                # Let C = N_v & M[b1] & M[b2]
                C = N_v & M[b1] & M[b2]
                
                # We need two distinct vertices a1, a2 in C such that a1 not connected to a2.
                # If C has at least 2 bits, let's check.
                # If all pairs in C are connected, then any a1, a2 would be connected.
                # So we need to find a pair in C that is NOT an edge.
                
                if C.bit_count() < 2:
                    continue
                
                # Iterate a1 in C
                a1_mask = C
                while a1_mask:
                    a1 = (a1_mask & -a1_mask).bit_length() - 1
                    a1_mask &= a1_mask - 1
                    
                    # a2 in C, a2 != a1, and a1 not connected to a2.
                    # a2 in C & ~(1 << a1) & ~M[a1]
                    a2_mask = C & ~(1 << a1) & ~M[a1]
                    if a2_mask:
                        # Found a K3,3
                        return False
        
        return True

    def add_edge(u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)

    def remove_edge(u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)

    def count_edges():
        return sum(M[i].bit_count() for i in range(n)) // 2

    def is_valid():
        # Check if current graph is K3,3 free
        # For all triples of vertices, check common neighbors count <= 2
        # This is O(n^3) or O(n^4) depending on implementation.
        # For n=20, n^3 = 8000. Fast enough.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (M[i] & M[j] & M[k]).bit_count() > 2:
                        return False
        return True

    best_edges = 0
    best_M = [0] * n

    # Strategy:
    # 1. Greedy construction with random order.
    # 2. Local search: try to add missing edges. If stuck, remove random edge and try again.
    # 3. Multiple restarts.

    random.seed(42)

    # Greedy construction
    for restart in range(100): # Limit restarts to avoid infinite loop if too slow
        if time.time() > deadline - 5:
            break
            
        M = [0] * n
        # Shuffle edge order for greedy
        edge_order = pairs[:]
        random.shuffle(edge_order)
        
        for u, v in edge_order:
            if can_add(u, v):
                add_edge(u, v)
        
        current_edges = count_edges()
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any missing edge
            # Shuffle missing edges to avoid bias
            missing = [(u, v) for u, v in pairs if not ((M[u] >> v) & 1)]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if can_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    improved = True
                    # Restart missing list? Or just continue.
                    # Continuing is faster.
            
            if not improved and current_edges > 0:
                # Perturbation: remove a random edge
                existing = [(u, v) for u, v in pairs if (M[u] >> v) & 1]
                if existing:
                    u, v = random.choice(existing)
                    remove_edge(u, v)
                    current_edges -= 1
                    # After removal, we might be able to add more
                    # The loop will restart if we set improved=True?
                    # No, we break out of the "try add" loop.
                    # We should set improved=True to loop again and try adding.
                    improved = True
                    # But we need to be careful not to oscillate.
                    # Simple approach: after removal, do another pass of adding.

        if current_edges > best_edges:
            best_edges = current_edges
            best_M = M[:]

    # Final check and return
    # Verify best_M is valid
    # (Assuming can_add is correct, it should be)
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
