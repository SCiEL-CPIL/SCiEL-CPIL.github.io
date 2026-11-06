import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n = 16
    
    def is_triangle_free_edges(edge_set):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            au = adj[u]
            for v in au:
                for w in au:
                    if w != v and w in adj[v]:
                        return False
        return True
    
    def is_triangle_free_mask(adj_mask):
        """adj_mask is a list of int bitmasks. Check triangle-free."""
        for u in range(n):
            au = adj_mask[u]
            for v in range(u+1, n):
                if (au >> v) & 1:
                    # Check if adj[u] & adj[v] has any common vertex w
                    if adj_mask[u] & adj_mask[v]:
                        return False
        return True
    
    def is_3_colorable_mask(adj_mask):
        """Check if graph with adjacency bitmasks is 3-colorable."""
        # Use bitset-based backtracking
        # For each vertex, try colors 0, 1, 2
        # Precompute neighbors
        color = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        # Map position in order to vertex
        pos_of = [0] * n
        for i, v in enumerate(order):
            pos_of[v] = i
        
        def dfs(pos):
            if pos == n:
                return True
            u = order[pos]
            for c in range(3):
                # Check if any neighbor already has color c
                # We need to check all neighbors v of u
                conflict = False
                v_mask = adj_mask[u]
                # Iterate over set bits in v_mask
                vm = v_mask
                while vm:
                    lowbit = vm & (-vm)
                    v = lowbit.bit_length() - 1
                    if color[v] == c:
                        conflict = True
                        break
                    vm ^= lowbit
                if not conflict:
                    color[u] = c
                    if dfs(pos + 1):
                        return True
                    color[u] = -1
            return False
        
        return dfs(0)
    
    def is_3_colorable_fast(adj_mask):
        """Faster 3-colorability check using bitset representation."""
        # color[i] stores the color of vertex i: -1 if uncolored, 0, 1, 2
        # For each color c, we maintain a bitmask of vertices colored c
        # A vertex u can take color c if (adj_mask[u] & colored[c]) == 0
        
        # Use bit-parallelism where possible
        colored = [0, 0, 0]  # bitmasks for colors 0, 1, 2
        color_of = [-1] * n
        
        # Order: try to color high-degree vertices first
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        def dfs(pos):
            if pos == n:
                return True
            u = order[pos]
            adj_u = adj_mask[u]
            for c in range(3):
                if (adj_u & colored[c]) == 0:
                    # Assign color c to u
                    colored[c] |= (1 << u)
                    color_of[u] = c
                    if dfs(pos + 1):
                        return True
                    colored[c] &= ~(1 << u)
                    color_of[u] = -1
            return False
        
        return dfs(0)
    
    def is_4_chromatic_mask(adj_mask):
        return not is_3_colorable_fast(adj_mask)
    
    def greedy_maximal_extension(adj_mask, deadline):
        """Given a triangle-free graph (as adj_mask), greedily add all possible edges."""
        adj_mask = list(adj_mask)
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj_mask[u] >> v) & 1):
                        # Check if adding edge (u,v) creates a triangle
                        if (adj_mask[u] & adj_mask[v]) == 0:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            improved = True
        return adj_mask
    
    def random_triangle_free_graph(deadline):
        """Generate a random triangle-free graph."""
        # Start with empty graph
        adj_mask = [0] * n
        # Add random edges, checking triangle-freeness
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if (adj_mask[u] & adj_mask[v]) == 0:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        return adj_mask
    
    def try_improve_by_swap(adj_mask, deadline):
        """Try to improve the edge count by removing one edge and adding up to two."""
        best_adj = list(adj_mask)
        best_count = sum(bin(m).count('1') for m in adj_mask) // 2
        
        # Get all edges
        edges = [(u, v) for u in range(n) for v in range(u+1, n) if (adj_mask[u] >> v) & 1]
        
        # For each edge, try removing it and then greedily adding back
        for u, v in edges:
            if time.time() > deadline - 5:
                break
            # Remove edge (u,v)
            new_adj = list(adj_mask)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Now greedily add all possible edges
            new_adj = greedy_maximal_extension(new_adj, deadline)
            
            new_count = sum(bin(m).count('1') for m in new_adj) // 2
            if new_count > best_count:
                best_adj = new_adj
                best_count = new_count
                # Continue searching from this improved state
                adj_mask = list(best_adj)
                edges = [(u, v) for u in range(n) for v in range(u+1, n) if (adj_mask[u] >> v) & 1]
        
        return best_adj, best_count
    
    def search_4_chromatic_deadline(deadline):
        """Main search: find 4-chromatic triangle-free graph with max edges."""
        global_best_adj = None
        global_best_count = -1
        
        # Strategy 1: Start from known 4-chromatic graphs and extend
        # Grötzsch graph (11 vertices, 20 edges, 4-chromatic)
        # Vertices 0-4 form C5, 5-9 form C5, vertex 10 connected to 5-9
        groetzsch_edges = [
            (0,1), (1,2), (2,3), (3,4), (4,0),  # C5 on 0-4
            (5,6), (6,7), (7,8), (8,9), (9,5),  # C5 on 5-9
            (0,5), (1,6), (2,7), (3,8), (4,9),  # Connections
            (5,10), (6,10), (7,10), (8,10), (9,10),  # 10 to 5-9
        ]
        
        groetzsch_adj = [0] * n
        for u, v in groetzsch_edges:
            groetzsch_adj[u] |= (1 << v)
            groetzsch_adj[v] |= (1 << u)
        
        # Extend Grötzsch graph
        extended = greedy_maximal_extension(groetzsch_adj, deadline)
        count = sum(bin(m).count('1') for m in extended) // 2
        if count > global_best_count and is_4_chromatic_mask(extended):
            global_best_adj = extended
            global_best_count = count
        
        # Try multiple random 4-chromatic cores
        # Core 1: Grötzsch + additional vertices
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            # Start from Grötzsch or random
            if trial % 2 == 0:
                adj_mask = list(groetzsch_adj)
                # Add random vertices
                for v in range(11, 16):
                    # Connect to random subset of existing vertices
                    neighbors = [u for u in range(11) if random.random() < 0.4]
                    for u in neighbors:
                        if (adj_mask[u] & adj_mask[v]) == 0:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
            else:
                # Random triangle-free graph
                adj_mask = random_triangle_free_graph(deadline)
            
            # Extend to maximal
            adj_mask = greedy_maximal_extension(adj_mask, deadline)
            
            # Check if 4-chromatic
            count = sum(bin(m).count('1') for m in adj_mask) // 2
            if count > global_best_count and is_4_chromatic_mask(adj_mask):
                global_best_adj = adj_mask
                global_best_count = count
        
        # Strategy 2: Local search from best found
        if global_best_adj is not None:
            if time.time() < deadline - 20:
                improved_adj, improved_count = try_improve_by_swap(global_best_adj, deadline)
                if improved_count > global_best_count and is_4_chromatic_mask(improved_adj):
                    global_best_adj = improved_adj
                    global_best_count = improved_count
        
        return global_best_adj, global_best_count
    
    adj_mask, count = search_4_chromatic_deadline(deadline)
    
    if adj_mask is not None:
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Fallback: return Grötzsch graph
    groetzsch_edges = [
        [0,1], [1,2], [2,3], [3,4], [4,0],
        [5,6], [6,7], [7,8], [8,9], [9,5],
        [0,5], [1,6], [2,7], [3,8], [4,9],
        [5,10], [6,10], [7,10], [8,10], [9,10],
    ]
    return groetzsch_edges

