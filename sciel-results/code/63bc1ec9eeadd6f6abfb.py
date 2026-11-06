"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    # Check common neighbors
                    if adj[i] & adj[j]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking with bitmasks."""
        # adj[i] is a bitmask of neighbors of i
        # Try to color vertices 0..15 with colors 0, 1, 2
        # color[v] = 0, 1, 2, or -1 if uncolored
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        order = sorted(range(16), key=lambda v: -degrees[v])
        order_idx = {v: i for i, v in enumerate(order)}
        
        # For each vertex, precompute which colors are forbidden by already-colored neighbors
        # We'll use a recursive backtracking approach
        
        # Precompute neighbor lists
        neighbors = [adj[i] for i in range(16)]
        
        # Use a stack-based approach
        # State: (vertex_index_in_order, color_assignment_bitmask)
        # color_assignment: for each vertex, 0=uncolored, 1=color0, 2=color1, 3=color2
        # Store as list of 16 values
        
        best = None
        
        def backtrack(pos, colors):
            """
            colors: list of 16 values, -1=uncolored, 0/1/2=colored
            pos: next vertex to color in order
            Returns True if 3-coloring found
            """
            nonlocal best
            if pos == 16:
                best = colors[:]
                return True
            
            v = order[pos]
            # Determine which colors are available
            available = [0, 1, 2]
            for c in available[:]:
                # Check if any neighbor has color c
                nmask = neighbors[v]
                for u in range(16):
                    if (nmask >> u) & 1:
                        if colors[u] == c:
                            available.remove(c)
                            break
            
            if not available:
                return False
            
            # Try colors in order of least constraining value first
            for c in available:
                colors[v] = c
                if backtrack(pos + 1, colors):
                    return True
                colors[v] = -1
            
            return False
        
        colors = [-1] * 16
        return backtrack(0, colors)
    
    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def edges_to_adj(edges):
        adj = [0] * 16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Start with a known 4-chromatic triangle-free graph
    # Mycielski of C5 (Grötzsch graph) on 11 vertices, plus 5 isolated
    # Let's use a better starting point - try to construct a denser one
    
    # Start with empty graph and build up
    # Actually, let's start with a random triangle-free graph and add edges
    
    def random_triangle_free_graph():
        """Generate a random triangle-free graph."""
        adj = [0] * 16
        # Try to add random edges while maintaining triangle-free
        all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding (u,v) creates a triangle
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if adj[u] & (1 << v):
            return False  # already exists
        common = adj[u] & adj[v]
        if common:
            return False  # would create triangle
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Generate initial solutions
    for _ in range(50):
        adj = random_triangle_free_graph()
        cnt = count_edges(adj)
        if cnt > best_count and not is_3_colorable(adj):
            best_adj = adj[:]
            best_count = cnt
    
    # If we didn't find a non-3-colorable graph, force one
    if best_adj is None:
        # Use the Grötzsch graph as starting point
        adj = [0] * 16
        # C5 on vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        # Mycielski construction: vertex 5 is center connected to 0-4
        for i in range(5):
            adj[5] |= (1 << i)
            adj[i] |= (1 << 5)
        # Vertices 6-10: copies of 0-4, connected to neighbors' copies and to 11
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[6 + i] |= (1 << (6 + j))
                    adj[6 + j] |= (1 << (6 + i))
            adj[6 + i] |= (1 << 11)
            adj[11] |= (1 << (6 + i))
        # Connect 11 to 5
        adj[11] |= (1 << 5)
        adj[5] |= (1 << 11)
        # Wait, standard Mycielski: M(G) has V(G) U V'(G) U {u}
        # where V'(G) = {v' : v in V(G)}, u connected to all V'(G)
        # v' connected to N(v) in V(G)
        
        # Let me redo this properly
        adj = [0] * 16
        # Original C5: vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            # Copy i is vertex 5+i
            # Connect 5+i to neighbors of i in original
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            # Connect 5+i to vertex 10
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        # This is the Grötzsch graph on 11 vertices
        # Verify it's triangle-free and 4-chromatic
        if is_triangle_free(adj) and not is_3_colorable(adj):
            best_adj = adj[:]
            best_count = count_edges(adj)
        else:
            # Fallback: just use it anyway
            best_adj = adj[:]
            best_count = count_edges(adj)
    
    # Phase 2: Local search to maximize edges
    if best_adj is not None:
        current_adj = best_adj[:]
        current_count = best_count
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Try to add a random edge
            all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                        if not (current_adj[i] & (1 << j))]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            improved = False
            
            for u, v in all_edges[:100]:  # Try up to 100 random edges
                # Check if adding (u,v) keeps triangle-free
                common = current_adj[u] & current_adj[v]
                if common:
                    continue
                
                # Add the edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Check if still not 3-colorable
                if not is_3_colorable(current_adj):
                    new_count = count_edges(current_adj)
                    if new_count > current_count:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = current_adj[:]
                        improved = True
                        break
                    else:
                        # Roll back
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                else:
                    # Roll back
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
            
            if not improved:
                no_improve_count += 1
                # Try removing an edge and adding two others (2-opt style)
                if no_improve_count % 10 == 0:
                    # Try a different move: remove one edge, add another
                    existing_edges = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                                    if current_adj[i] & (1 << j)]
                    if existing_edges:
                        random.shuffle(existing_edges)
                        for u, v in existing_edges[:50]:
                            # Remove (u,v)
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                            
                            # Try to add a different edge
                            other_edges = [(a, b) for a in range(16) for b in range(a + 1, 16) 
                                          if not (current_adj[a] & (1 << b))]
                            random.shuffle(other_edges)
                            found = False
                            for a, b in other_edges[:50]:
                                common = current_adj[a] & current_adj[b]
                                if common:
                                    continue
                                current_adj[a] |= (1 << b)
                                current_adj[b] |= (1 << a)
                                if not is_3_colorable(current_adj):
                                    new_count = count_edges(current_adj)
                                    if new_count > current_count:
                                        current_count = new_count
                                        if current_count > best_count:
                                            best_count = current_count
                                            best_adj = current_adj[:]
                                        found = True
                                        break
                                    else:
                                        current_adj[a] &= ~(1 << b)
                                        current_adj[b] &= ~(1 << a)
                                else:
                                    current_adj[a] &= ~(1 << b)
                                    current_adj[b] &= ~(1 << a)
                            
                            if not found:
                                # Restore removed edge
                                current_adj[u] |= (1 << v)
                                current_adj[v] |= (1 << u)
                            else:
                                no_improve_count = 0
                                break
            else:
                no_improve_count = 0
        
        # Phase 3: More aggressive search with restarts
        if time.time() < deadline:
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                # Start from a perturbed version of best_adj
                adj = best_adj[:]
                # Remove some random edges
                existing = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                           if adj[i] & (1 << j)]
                random.shuffle(existing)
                for u, v in existing[:random.randint(0, 5)]:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                
                # Try to add edges greedily
                all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16) 
                               if not (adj[i] & (1 << j))]
                random.shuffle(all_possible)
                
                for u, v in all_possible:
                    if time.time() > deadline:
                        break
                    common = adj[u] & adj[v]
                    if common:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not is_3_colorable(adj):
                        cnt = count_edges(adj)
                        if cnt > best_count:
                            best_count = cnt
                            best_adj = adj[:]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
    
    if best_adj is None:
        # Absolute fallback: return the seed graph
        edges = []
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        for u, w in [(i, (i + 1) % 5) for i in range(5)]:
            edges.append([u, 5 + w])
            edges.append([w, 5 + u])
        for u in range(5):
            edges.append([5 + u, 10])
        return edges
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
