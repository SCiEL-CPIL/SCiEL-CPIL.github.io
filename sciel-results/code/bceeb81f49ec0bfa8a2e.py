"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Precompute triangle conflicts using bitsets
    # For each pair of vertices (u,v), compute which other vertices form triangles
    # Triangle (u,v,w) exists if w is adjacent to both u and v
    # We need: for edge (u,v), the set of edges that would create a triangle with it
    # Edge (u,v) conflicts with edge (w,x) if {u,v} ∩ {w,x} != empty AND the third vertex creates triangle
    
    # Better: for each potential edge (u,v), precompute which other edges (a,b) would form a triangle
    # Edge (u,v) + edge (w,x) forms triangle iff:
    #   - They share exactly one vertex, say u=w, and v-x is an edge, OR
    #   - They share exactly one vertex, say u=x, and v-w is an edge, OR  
    #   - They share exactly one vertex, say v=w, and u-x is an edge, OR
    #   - They share exactly one vertex, say v=x, and u-w is an edge
    # Actually simpler: two edges (u,v) and (a,b) are incompatible (form triangle) iff
    #   they share exactly one vertex and the other two vertices are connected
    # i.e., if u==a, then v-b must be an edge; if u==b, then v-a must be an edge; etc.
    
    # Map each edge to an index
    edge_to_idx = {}
    idx_to_edge = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(idx_to_edge)
            idx_to_edge.append((u, v))
    
    # For each edge, compute bitmask of conflicting edges
    # Two edges conflict if they share exactly one vertex and the remaining vertices are adjacent
    conflict_mask = [0] * total_possible_edges
    
    for i in range(total_possible_edges):
        u1, v1 = idx_to_edge[i]
        mask = 0
        for j in range(i + 1, total_possible_edges):
            u2, v2 = idx_to_edge[j]
            # Check if they share exactly one vertex
            common = 0
            if u1 == u2 or u1 == v2:
                common += 1
            if v1 == u2 or v1 == v2:
                common += 1
            if common == 1:
                # They share exactly one vertex, check if the other two form an edge
                # Get the two non-shared vertices
                if u1 == u2:
                    a, b = v1, v2
                elif u1 == v2:
                    a, b = v1, u2
                elif v1 == u2:
                    a, b = u1, v2
                else:  # v1 == v2
                    a, b = u1, u2
                # Check if (a,b) is a potential edge (always true for a != b in K16)
                # But we need to check if adding both edges would create a triangle
                # The triangle would be (shared_vertex, a, b) if a-b is an edge
                # Since we're building the conflict graph for the line graph of K16,
                # two edges conflict if they can't both be in a triangle-free set
                # They can't both be in a triangle-free set iff they share a vertex AND the other endpoints are adjacent
                # In K16, all vertices are adjacent, so ANY two edges sharing a vertex conflict!
                # Wait, that's not right. In a triangle-free graph, if edges (u,v) and (u,w) are both present,
                # then v-w must NOT be an edge. But v-w could be an edge or not.
                # The conflict is: if we have edges (u,v) and (u,w), we can only have both if (v,w) is NOT in our edge set.
                # So the conflict depends on whether (v,w) is also selected.
                # This means we can't precompute simple pairwise conflicts - it's a 3-way constraint.
                
                # Actually, let me reconsider. The problem is: find maximum subset of edges from K16 such that
                # no three edges form a triangle. This is equivalent to finding the maximum independent set
                # in the 3-uniform hypergraph where each hyperedge is a triangle.
                
                # For branch-and-bound, we can use: at each step, when we add an edge, we must ensure
                # no triangle is completed. We can track which triangles are "blocked" or "completed".
                
                pass
        conflict_mask[i] = mask
    
    # The above approach doesn't work well because it's a 3-uniform constraint.
    # Let me use a different approach: backtracking with bitset representation of the graph.
    
    # Represent the graph as a bitset for each vertex
    # adj[v] is a bitmask of vertices adjacent to v
    
    # Start with the Grötzsch graph as seed
    def get_groetzsch_adj():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        for i in range(5):
            j = (i + 1) % 5
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        # Mycielski: 5..9
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for n_idx in neighbors:
                adj[5 + i] |= (1 << n_idx)
                adj[n_idx] |= (1 << (5 + i))
        # Vertex 10
        for i in range(5):
            adj[10] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
        return adj
    
    # Convert adj bitmasks to edge list
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask &= mask - 1
        return edges
    
    # Check if graph is 3-colorable using backtracking
    def is_3_colorable(adj):
        color = [-1] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = set()
            mask = adj[v]
            while mask:
                lsb = mask & (-mask)
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used.add(color[u])
                mask &= mask - 1
            
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Branch and bound to find maximum triangle-free edge set
    # State: current adjacency (list of bitmasks), current edge count
    # We try to add edges one by one, using branching and pruning
    
    best_edge_count = 0
    best_adj = None
    
    # Initialize with Grötzsch graph
    init_adj = get_groetzsch_adj()
    init_edges = adj_to_edges(init_adj)
    if not is_3_colorable(init_adj):
        best_edge_count = len(init_edges)
        best_adj = list(init_adj)
    
    # Precompute all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # For each edge, precompute which vertices it connects
    # and which existing edges would conflict (create triangle)
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without creating a triangle."""
        # Triangle would be formed if there exists w such that (u,w) and (v,w) are both in adj
        common_neighbors = adj[u] & adj[v]
        return common_neighbors == 0
    
    def add_edge_to_adj(adj, u, v):
        """Add edge (u,v) to adjacency list (mutates adj)."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge_from_adj(adj, u, v):
        """Remove edge (u,v) from adjacency list (mutates adj)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Branch and bound
    # We process edges in some order, deciding to include or exclude each
    # Prune if even adding all remaining possible edges can't beat best
    
    # Sort edges to process in a good order (maybe by how restrictive they are)
    # Edges between high-degree vertices are more restrictive
    
    # For efficiency, we'll use a recursive search with memoization-like pruning
    # But the state space is huge, so we need good heuristics
    
    # Alternative: use local search with bitsets for speed
    
    # Let me try a hybrid: start from Grötzsch, then do intensive local search
    # with bitset operations for speed
    
    current_adj = list(init_adj)
    current_edges = len(init_edges)
    
    # Phase 1: Greedy edge addition
    def greedy_add_edges(adj, max_iterations=1000):
        """Try to add edges greedily."""
        added = 0
        for _ in range(max_iterations):
            progress = False
            # Try adding each possible edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge_to_adj(adj, u, v)
                            added += 1
                            progress = True
            if not progress:
                break
        return added
    
    # Phase 2: Local search with edge swaps
    def local_search(adj, num_iterations=10000):
        """Try to improve by removing one edge and adding one or two others."""
        best_adj_local = list(adj)
        best_count = sum(bin(a).count('1') for a in adj) // 2
        
        for iter_num in range(num_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Pick a random edge to remove
            edge_list = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    if v > u:
                        edge_list.append((u, v))
                    mask &= mask - 1
            
            if not edge_list:
                break
            
            u1, v1 = random.choice(edge_list)
            
            # Remove it
            remove_edge_from_adj(adj, u1, v1)
            
            # Try to add back as many edges as possible
            added_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge_to_adj(adj, u, v)
                            added_edges.append((u, v))
            
            current_count = sum(bin(a).count('1') for a in adj) // 2
            
            # Check 3-colorability
            if not is_3_colorable(adj):
                if current_count > best_count:
                    best_count = current_count
                    best_adj_local = list(adj)
            else:
                # Revert
                remove_edge_from_adj(adj, u1, v1)
                for u, v in added_edges:
                    remove_edge_from_adj(adj, u, v)
                add_edge_to_adj(adj, u1, v1)
        
        return best_adj_local, best_count
    
    # Run greedy addition
    added = greedy_add_edges(current_adj)
    if added > 0:
        current_edges += added
        if not is_3_colorable(current_adj):
            if current_edges > best_edge_count:
                best_edge_count = current_edges
                best_adj = list(current_adj)
    
    # Run local search
    best_adj_ls, best_count_ls = local_search(current_adj, num_iterations=50000)
    if best_count_ls > best_edge_count:
        best_edge_count = best_count_ls
        best_adj = best_adj_ls
    
    # Phase 3: More aggressive search - try multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 15.0:
            break
        
        # Start from a random perturbation of the best solution
        perturbed_adj = list(best_adj)
        
        # Remove some random edges
        num_removals = random.randint(2, 5)
        for _ in range(num_removals):
            edge_list = []
            for u in range(n):
                mask = perturbed_adj[u]
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    if v > u:
                        edge_list.append((u, v))
                    mask &= mask - 1
            if not edge_list:
                break
            u, v = random.choice(edge_list)
            remove_edge_from_adj(perturbed_adj, u, v)
        
        # Greedy add
        added = greedy_add_edges(perturbed_adj)
        current_count = sum(bin(a).count('1') for a in perturbed_adj) // 2
        
        # Local search
        best_adj_rs, best_count_rs = local_search(perturbed_adj, num_iterations=20000)
        if best_count_rs > best_edge_count:
            best_edge_count = best_count_rs
            best_adj = best_adj_rs
    
    # Final: ensure best_adj is valid
    if best_adj is None:
        best_adj = init_adj
        best_edge_count = len(init_edges)
    
    # Convert to edge list
    result_edges = adj_to_edges(best_adj)
    
    return result_edges
# EVOLVE-BLOCK-END
