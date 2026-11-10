import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Grotzsch graph: 11 vertices, 20 edges, triangle-free, 4-chromatic
    # This is the Mycielski graph of C5
    grotzsch_edges = []
    # Original C5
    for i in range(5):
        grotzsch_edges.append([i, (i+1)%5])
    # Mycielski vertices 5..9
    grotzsch_edges.append([5, 1])
    grotzsch_edges.append([5, 4])
    grotzsch_edges.append([6, 0])
    grotzsch_edges.append([6, 2])
    grotzsch_edges.append([7, 1])
    grotzsch_edges.append([7, 3])
    grotzsch_edges.append([8, 2])
    grotzsch_edges.append([8, 4])
    grotzsch_edges.append([9, 3])
    grotzsch_edges.append([9, 0])
    # 10 connected to 5,6,7,8,9
    for i in range(5):
        grotzsch_edges.append([10, 5+i])
    
    # Build adjacency matrix for Grotzsch
    base_adj = [[0]*n for _ in range(n)]
    for u, v in grotzsch_edges:
        base_adj[u][v] = 1
        base_adj[v][u] = 1
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def is_3_colorable(adj):
        """
        Fast backtracking 3-coloring with bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        # Precompute adjacency as bitmasks for speed
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        
        color = [-1] * n
        uncolored = list(range(n))
        
        def can_color():
            if not uncolored:
                return True
            
            # Pick uncolored vertex with fewest available colors (MRV heuristic)
            best_idx = -1
            best_count = 100
            best_avail = 0
            for idx, v in enumerate(uncolored):
                # Compute forbidden colors from colored neighbors
                forbidden = 0
                mask = adj_mask[v]
                for u in range(n):
                    if color[u] != -1 and (mask >> u) & 1:
                        forbidden |= (1 << color[u])
                avail = 7 & ~forbidden  # 7 = 0b111, available colors
                count = bin(avail).count('1')
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_idx = idx
                    best_avail = avail
                    if count == 1:
                        break
            
            v = uncolored[best_idx]
            uncolored.pop(best_idx)
            
            # Try each available color
            avail = best_avail
            while avail:
                c_bit = avail & (-avail)  # lowest set bit
                avail ^= c_bit
                c = c_bit.bit_length() - 1
                color[v] = c
                if can_color():
                    return True
                color[v] = -1
            
            uncolored.insert(best_idx, v)
            return False
        
        return can_color()
    
    def edge_count(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    # Strategy: Start with Grotzsch + 5 isolated vertices, then try to add edges
    # The 5 isolated vertices can connect to vertices in Grotzsch
    # We need to maintain triangle-freeness and 4-chromaticity
    
    # Key insight: The Grotzsch graph has a maximum independent set of size 5
    # If we connect the 5 new vertices to this independent set, we add 25 edges
    # and maintain triangle-freeness (since independent set vertices have no edges between them)
    
    # Find maximum independent set in Grotzsch
    def find_max_independent_set(adj, vertices):
        """Find maximum independent set using backtracking."""
        best_set = []
        
        def backtrack(remaining, current_set):
            nonlocal best_set
            if len(current_set) >= len(best_set):
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
            
            if not remaining:
                return
            
            # Try adding the first vertex
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include v
            new_remaining = [u for u in rest if not adj[v][u]]
            backtrack(new_remaining, current_set + [v])
            
            # Option 2: Exclude v
            backtrack(rest, current_set)
        
        backtrack(list(vertices), [])
        return best_set
    
    # Grotzsch vertices are 0..10
    grotzsch_vertices = list(range(11))
    max_iset = find_max_independent_set(base_adj, grotzsch_vertices)
    
    # The 5 new vertices are 11..15
    new_vertices = list(range(11, 16))
    
    # Connect each new vertex to the maximum independent set
    # This adds 5*5 = 25 edges
    adj = [row[:] for row in base_adj]
    for nv in new_vertices:
        for iv in max_iset:
            adj[nv][iv] = 1
            adj[iv][nv] = 1
    
    # Check if this is valid
    if is_triangle_free(adj) and not is_3_colorable(adj):
        m = edge_count(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Now try to improve by adding more edges
    # Try all possible edges between new vertices and Grotzsch vertices
    # and between new vertices themselves
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v) while maintaining triangle-freeness and 4-chromaticity."""
        if adj[u][v]:
            return adj
        
        # Check triangle-freeness using common neighbors
        # Two vertices can be connected iff they have no common neighbor
        common = 0
        for k in range(n):
            if k != u and k != v and adj[u][k] and adj[v][k]:
                common = 1
                break
        if common:
            return adj
        
        # Try adding
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Check 4-chromaticity
        if not is_3_colorable(adj):
            return adj
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            return adj
    
    # Precompute list of all non-edges for faster iteration
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        return non_edges
    
    # Get all edges
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges
    
    # Try edge swap: remove edge (u,v), add edge (a,b)
    def try_edge_swap(adj, remove_e, add_e):
        """Try swapping one edge for another while maintaining constraints."""
        u, v = remove_e
        a, b = add_e
        
        # Check if add_e is already present
        if adj[a][b]:
            return adj
        
        # Check if remove_e is present
        if not adj[u][v]:
            return adj
        
        # Temporarily remove edge (u,v)
        adj[u][v] = 0
        adj[v][u] = 0
        
        # Check if we can add edge (a,b)
        if not adj[a][b]:
            # Check triangle-freeness
            common = 0
            for k in range(n):
                if k != a and k != b and adj[a][k] and adj[b][k]:
                    common = 1
                    break
            if not common:
                adj[a][b] = 1
                adj[b][a] = 1
                
                # Check 4-chromaticity
                if not is_3_colorable(adj):
                    return adj
                else:
                    # Revert
                    adj[a][b] = 0
                    adj[b][a] = 0
        
        # Revert removal
        adj[u][v] = 1
        adj[v][u] = 1
        return adj
    
    # Iteratively try to add edges, prioritizing edges that are "close" to being addable
    improved = True
    while improved and time.time() < deadline:
        improved = False
        non_edges = get_non_edges(adj)
        # Shuffle to avoid deterministic ordering bias
        random.shuffle(non_edges)
        for i, j in non_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:  # May have been added by a previous iteration
                continue
            new_adj = try_add_edge(adj, i, j)
            if new_adj is not adj:
                adj = new_adj
                improved = True
        
        # Try edge swaps to escape local optima
        if time.time() < deadline:
            edges = get_edges(adj)
            non_edges = get_non_edges(adj)
            random.shuffle(edges)
            random.shuffle(non_edges)
            for rem_e in edges[:20]:  # Limit swaps per iteration
                if time.time() > deadline:
                    break
                for add_e in non_edges[:20]:
                    if time.time() > deadline:
                        break
                    if adj[add_e[0]][add_e[1]]:
                        continue
                    new_adj = try_edge_swap(adj, rem_e, add_e)
                    if new_adj is not adj:
                        adj = new_adj
                        improved = True
                        break
        
        m = edge_count(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # If we didn't find anything better, try random search
    if best_score < 45:
        random.seed(42)
        
        def random_graph(density):
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < density:
                        # Check if adding (i,j) creates a triangle
                        ok = True
                        for k in range(n):
                            if k != i and k != j:
                                if adj[i][k] and adj[j][k]:
                                    ok = False
                                    break
                        if ok:
                            adj[i][j] = 1
                            adj[j][i] = 1
            return adj
        
        for restart in range(100):
            if time.time() > deadline:
                break
            
            density = 0.3 + 0.3 * (restart % 5) / 4
            adj = random_graph(density)
            
            if not is_triangle_free(adj):
                continue
            
            if is_3_colorable(adj):
                continue
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                non_edges = get_non_edges(adj)
                random.shuffle(non_edges)
                for i, j in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[i][j]:
                        continue
                    new_adj = try_add_edge(adj, i, j)
                    if new_adj is not adj:
                        adj = new_adj
                        improved = True
                m = edge_count(adj)
                if m > best_score:
                    best_score = m
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Fallback: return Grotzsch + isolated
    if best_edges is None:
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if base_adj[i][j]]
    
    return best_edges
