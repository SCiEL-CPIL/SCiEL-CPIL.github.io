import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses bipartite double-cover representation with bulk Boolean matrix
    multiplication to detect forbidden C6 configurations efficiently.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    # Build adjacency matrix from edge list
    def build_adj_matrix(edges):
        A = np.zeros((N, N), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A
    
    # Check if adding edge (u, v) creates a C6 using the bipartite double-cover.
    # In the double cover, we have two copies: left (0..39) and right (40..79).
    # An edge (u,v) in original becomes edges (u_L, v_R) and (v_L, u_R) in cover.
    # A C6 in original corresponds to a C4 in the cover.
    # We check: does there exist a path u_L -> x_R -> y_L -> z_R -> w_L -> v_R
    # where all intermediate nodes are distinct from u and v?
    # Equivalently: count paths of length 4 from u to v in original avoiding u,v at ends.
    # Use matrix powers: (A^4)[u,v] counts walks of length 4, but we need simple paths.
    # For efficiency, we use a BFS-based check with early termination.
    
    def can_add_fast(adj_list, adj_set, u, v):
        """Check if adding (u,v) creates a C6 using BFS with pruning."""
        # Find all nodes at distance exactly 4 from u, avoiding v
        # Then check if any such node is adjacent to v
        
        # BFS from u to depth 4, tracking visited to ensure simple path
        # We'll use a set of (node, depth, frozenset_of_intermediate_nodes)
        # But that's expensive. Instead, use a smarter approach:
        
        # A C6 exists iff there's a path u-a-b-c-d-v where {a,b,c,d} are distinct
        # and none equal to u or v.
        
        # Approach: For each common structure, check if we can form the cycle.
        # Use the fact that in a C6-free graph, for any non-adjacent u,v,
        # the number of length-4 paths between them must be 0.
        
        # Efficient check: 
        # Step 1: Get neighbors of u (excluding v)
        # Step 2: Get neighbors of v (excluding u) 
        # Step 3: For each pair (a in N(u), d in N(v)), check if there's
        #         a path a-b-c-d of length 2 (i.e., common neighbor of a and d's neighbors)
        #         Actually, we need a-b-c-d where b,c are intermediate.
        
        # Let's use a different approach: 
        # Count the number of 4-paths from u to v. If > 0, C6 exists.
        # A 4-path is u-a-b-c-d-v where all 6 nodes distinct.
        
        # Use matrix approach for bulk, but for single edge check, use BFS with memoization.
        
        # Simple BFS approach with depth limit
        from collections import deque
        
        # We want paths of length 4 from u to v (i.e., 5 edges total including u-v)
        # Wait, C6 has 6 edges: u-a, a-b, b-c, c-d, d-v, and v-u (the new edge)
        # So we need a path of length 4 from u to v: u-a-b-c-d-v
        
        # BFS from u, tracking path nodes to ensure simplicity
        # For efficiency, limit to depth 4
        
        # Use a dictionary: node -> set of frozensets of intermediate nodes
        # This can blow up, so use a heuristic: only track up to a few paths per node
        
        # Alternative: use the adjacency matrix power trick
        # (A^4)[u][v] counts walks of length 4 from u to v
        # But walks can repeat nodes. For a sparse graph, most walks are simple.
        # We can use this as a quick filter: if (A^4)[u][v] == 0, definitely no C6.
        # If > 0, we need to verify with exact check.
        
        # For now, use a direct BFS check
        # State: (current_node, depth, visited_mask)
        # visited_mask is a 40-bit integer
        
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if v is adjacent to node and v not in path
                if v in adj_set[node] and not (mask & (1 << v)):
                    return False  # C6 exists
            
            if depth >= 4:
                continue
            
            for nb in adj_list[node]:
                if nb == v:
                    continue
                if mask & (1 << nb):
                    continue
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return True
    
    # Bulk check: given current edge set, find all pairs (u,v) that can be added
    # without creating a C6. Uses the observation that we can precompute
    # distance-4 reachability.
    
    def find_addable_edges(adj_list, adj_set, all_pairs):
        """Find all missing edges that can be added without creating a C6."""
        addable = []
        for u, v in all_pairs:
            if v in adj_set[u]:
                continue
            if can_add_fast(adj_list, adj_set, u, v):
                addable.append((u, v))
        return addable
    
    # Initial Graph: 8 disjoint K5s
    edges = []
    adj_list = [[] for _ in range(N)]
    adj_set = [set() for _ in range(N)]
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append((u, v))
                
    best_edges = len(edges)
    best_edges_copy = edges[:]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local Search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Find all addable edges
        addable = find_addable_edges(adj_list, adj_set, all_pairs)
        
        if not addable:
            # Stuck. Perform a kick
            if len(edges) > 0:
                # Remove 1 to 5 random edges for more aggressive perturbation
                kick_size = random.randint(1, 5)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    # Remove from structures
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                    edges.pop(idx)
            continue
        
        # Sort by degree sum (prefer connecting high-degree vertices)
        addable.sort(key=lambda x: -(len(adj_list[x[0]]) + len(adj_list[x[1]])))
        
        added_in_pass = 0
        for u, v in addable:
            if time.time() > deadline:
                break
            if v in adj_set[u]:
                continue
            # Re-check since graph may have changed
            if can_add_fast(adj_list, adj_set, u, v):
                # Add edge
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append((u, v))
                added_in_pass += 1
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_edges_copy = edges[:]
        
        if added_in_pass == 0:
            # Stuck. Perform a kick
            if len(edges) > 0:
                kick_size = random.randint(1, 5)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                    edges.pop(idx)

    # Return the best graph found
    final_edges = [[u, v] for u, v in best_edges_copy]
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
