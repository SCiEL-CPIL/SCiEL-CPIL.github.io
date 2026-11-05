import time
import random
import sys

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    # Bitset representation of adjacency
    # adj[i] is an integer where bit j is 1 if edge (i, j) exists
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def get_neighbors(u):
        # Return list of neighbors
        mask = adj[u]
        neighbors = []
        i = 0
        while mask:
            if mask & 1:
                neighbors.append(i)
            mask >>= 1
            i += 1
        return neighbors
        
    def can_add(u, v):
        """
        Returns True if adding edge (u, v) does not create a C6.
        This requires that there is NO path of length 5 between u and v in the current graph.
        """
        if has_edge(u, v):
            return False
            
        # Calculate reachable vertices in exactly k steps from u
        # Using bitsets for speed
        # R_1 = N(u)
        r = adj[u]
        
        # R_2: vertices reachable in 2 steps
        # r is a bitset. We need union of N(x) for all x in r.
        # Optimization: If r is empty, return False (no path).
        # Loop through bits in r.
        curr = r
        next_r = 0
        # We need N_2, N_3, N_4, N_5.
        # N_5(u) contains v?
        
        # Step 1: N_1(u) = r
        # Step 2: N_2(u) = Union(N(x) for x in N_1(u))
        # To avoid counting paths that backtrack, we technically need simple paths?
        # No, "path of length 5" in graph theory for cycle detection usually implies distinct vertices 
        # for the cycle to be a C6. 
        # However, if there is a walk of length 5, does it imply a simple path?
        # Not necessarily. But if there is a C6, there is a simple path of length 5.
        # If we find a walk of length 5, it might be $u-a-b-a-c-v$. This is not a C6.
        # But checking simple paths is hard.
        # However, for C6-free graphs, a necessary and sufficient condition is often checked via
        # matrix powers, which count walks.
        # If (A^5)[u,v] > 0, there is a walk of length 5.
        # Does a walk of length 5 imply a C6?
        # If the walk is $u-x-y-z-w-v$, and vertices are distinct, it's a C6.
        # If not distinct, e.g., $u-x-y-x-z-v$, it's a shorter structure.
        # But if the graph is C6-free, can it have a walk of length 5?
        # Yes, e.g., in a K5, there are many walks of length 5. But K5 is C6-free.
        # So $A^5[u,v] > 0$ is NOT a sufficient condition for C6 existence?
        # Wait. K5 has no C6. But it has walks of length 5.
        # So the condition "A^5[u,v] == 0" is sufficient for safety (no walk => no path => no C6),
        # but not necessary?
        # No, if we add (u,v) to K5, we create K6, which is C6-free.
        # So we CAN add edges even if A^5 > 0.
        
        # The correct condition: Adding (u,v) creates a C6 if and only if there is a SIMPLE path of length 5.
        # Checking for simple paths is NP-hard in general, but for n=40 and specific local structure, 
        # we can do a bounded DFS/BFS.
        
        # Let's do a BFS/DFS from u to find if v is reachable in exactly 5 steps with distinct vertices.
        # Since degree is small, this is fast.
        
        # State: (current_node, visited_mask)
        # Depth 5.
        # Branching factor ~5-10. 10^5 is 100,000. Might be too slow in Python for many checks.
        
        # Let's optimize:
        # Only check if u and v are "close" in some metric?
        # Or just do the BFS.
        
        # Alternative: Use the property that C6-free graphs have restricted structure.
        # But let's stick to a fast BFS for depth 5.
        
        # visited[depth][node] ? No, need to track visited set.
        # We can use a set of visited nodes for each path.
        
        # Since we only need to know IF a path exists, we can prune aggressively.
        
        # Let's try a different approach:
        # If we add (u,v), the new cycle is u-v-...-u.
        # The path ... must be simple.
        
        # Let's implement a fast check for simple path of length 5.
        # From u, expand neighbors.
        # Depth 0: {u}
        # Depth 1: N(u)
        # Depth 2: N(N(u)) \ {u, N(u)} ? No, just N(N(u)) excluding visited.
        
        # We will maintain a list of (node, visited_mask).
        # Max size of this list at depth 3 is ~ deg(u)*deg(v)*deg(w).
        
        curr_states = [(u, 1 << u)]
        for depth in range(1, 5):
            next_states = []
            for node, mask in curr_states:
                neighbors = get_neighbors(node)
                for nb in neighbors:
                    if (mask >> nb) & 1:
                        continue
                    # If we are at depth 4 (going to depth 5), we check if nb == v
                    if depth == 4:
                        if nb == v:
                            return False # Path found, cannot add
                    else:
                        next_states.append((nb, mask | (1 << nb)))
            curr_states = next_states
            if not curr_states:
                break
                
        return True # No path of length 5 found
        
    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2
        
    # Initialization: Start with 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(base + a, base + b)
                
    best_edges = count_edges()
    best_adj = adj[:]
    
    # Greedy Add Pass
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Try to add random edges
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if has_edge(u, v):
                continue
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                if count_edges() > best_edges:
                    best_edges = count_edges()
                    best_adj = adj[:]
        
        if not improved:
            # Local Search: Remove a random edge, try to add two others
            edges_in_graph = []
            for u in range(n):
                for v in range(u + 1, n):
                    if has_edge(u, v):
                        edges_in_graph.append((u, v))
                        
            if not edges_in_graph:
                break
                
            # Pick a random edge to remove
            ru, rv = random.choice(edges_in_graph)
            
            # Save state
            old_adj = adj[:]
            
            remove_edge(ru, rv)
            
            # Try to add random edges to gain net +1
            added = 0
            attempts = 0
            max_attempts = 200
            while added < 2 and attempts < max_attempts and time.time() < deadline:
                attempts += 1
                nu, nv = random.choice(all_pairs)
                if has_edge(nu, nv):
                    continue
                if can_add(nu, nv):
                    add_edge(nu, nv)
                    added += 1
                    
            if count_edges() > best_edges:
                best_edges = count_edges()
                best_adj = adj[:]
            else:
                # Revert
                adj = old_adj[:]
                
    # Return best result
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# Keep the function name exactly as construct
