import time
import random

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Grötzsch graph edges (vertices 0-10)
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    def make_adj(edge_list):
        adj = [[0]*n for _ in range(n)]
        for u, v in edge_list:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def count_edges(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    def is_triangle_free_add(adj, u, v):
        """Check if adding edge (u,v) creates a triangle."""
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    return False
        return True
    
    def is_3_colorable(adj):
        """
        Fast backtracking 3-coloring using bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        # Build adjacency masks
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
        
        color = [-1] * n
        
        def backtrack(uncolored):
            if not uncolored:
                return True
            
            # MRV: pick vertex with fewest available colors
            best_idx = -1
            best_count = 100
            best_avail = 0
            for idx, v in enumerate(uncolored):
                forbidden = 0
                mask = adj_mask[v]
                for u in range(n):
                    if color[u] != -1 and (mask >> u) & 1:
                        forbidden |= (1 << color[u])
                avail = 7 & ~forbidden
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
            new_uncolored = uncolored[:best_idx] + uncolored[best_idx+1:]
            
            avail = best_avail
            while avail:
                c_bit = avail & (-avail)
                avail ^= c_bit
                c = c_bit.bit_length() - 1
                color[v] = c
                if backtrack(new_uncolored):
                    return True
                color[v] = -1
            
            return False
        
        return backtrack(list(range(n)))
    
    def greedy_fill(adj):
        """Greedily add all possible edges that don't create triangles."""
        new_adj = [row[:] for row in adj]
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not new_adj[i][j]:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if is_triangle_free_add(new_adj, u, v):
                new_adj[u][v] = 1
                new_adj[v][u] = 1
                
        return new_adj
    
    def local_search(adj, max_iter=5000):
        """
        Local search to add edges while maintaining triangle-freeness and non-3-colorability.
        Uses edge addition and edge swap moves.
        """
        adj = [row[:] for row in adj]
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Get non-edges
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        non_edges.append((i, j))
            
            random.shuffle(non_edges)
            
            # Try adding edges
            for u, v in non_edges:
                if time.time() > deadline:
                    return adj
                
                if not is_triangle_free_add(adj, u, v):
                    continue
                
                # Tentatively add
                adj[u][v] = 1
                adj[v][u] = 1
                
                if not is_3_colorable(adj):
                    improved = True
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
            
            # Try edge swaps: remove one edge, add another
            if not improved:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append((i, j))
                
                random.shuffle(edges)
                
                for u, v in edges:
                    if time.time() > deadline:
                        return adj
                    
                    # Remove edge (u,v)
                    adj[u][v] = 0
                    adj[v][u] = 0
                    
                    # Try to add a different edge
                    for a, b in non_edges:
                        if (a, b) == (u, v) or (a, b) == (v, u):
                            continue
                        if not is_triangle_free_add(adj, a, b):
                            continue
                        
                        adj[a][b] = 1
                        adj[b][a] = 1
                        
                        if not is_3_colorable(adj):
                            # Good swap
                            adj[u][v] = 1
                            adj[v][u] = 1
                            improved = True
                            break
                        
                        adj[a][b] = 0
                        adj[b][a] = 0
                    
                    # Restore original edge
                    adj[u][v] = 1
                    adj[v][u] = 1
            
            # Try 2-edge swaps
            if not improved:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append((i, j))
                
                if len(edges) >= 2:
                    random.shuffle(edges)
                    for i in range(min(len(edges), 20)):
                        if time.time() > deadline:
                            return adj
                        
                        e1 = edges[i]
                        for j in range(i+1, min(len(edges), 20)):
                            e2 = edges[j]
                            
                            # Remove both edges
                            adj[e1[0]][e1[1]] = 0
                            adj[e1[1]][e1[0]] = 0
                            adj[e2[0]][e2[1]] = 0
                            adj[e2[1]][e2[0]] = 0
                            
                            # Try adding two new edges
                            for a, b in non_edges:
                                if (a, b) in [(e1[0], e1[1]), (e2[0], e2[1])]:
                                    continue
                                if not is_triangle_free_add(adj, a, b):
                                    continue
                                
                                adj[a][b] = 1
                                adj[b][a] = 1
                                
                                for c, d in non_edges:
                                    if (c, d) in [(a, b), (e1[0], e1[1]), (e2[0], e2[1])]:
                                        continue
                                    if not is_triangle_free_add(adj, c, d):
                                        continue
                                    
                                    adj[c][d] = 1
                                    adj[d][c] = 1
                                    
                                    if not is_3_colorable(adj):
                                        # Restore original edges
                                        adj[e1[0]][e1[1]] = 1
                                        adj[e1[1]][e1[0]] = 1
                                        adj[e2[0]][e2[1]] = 1
                                        adj[e2[1]][e2[0]] = 1
                                        improved = True
                                        break
                                    
                                    adj[c][d] = 0
                                    adj[d][c] = 0
                                
                                if improved:
                                    break
                                
                                adj[a][b] = 0
                                adj[b][a] = 0
                            
                            if improved:
                                break
                            
                            # Restore
                            adj[e1[0]][e1[1]] = 1
                            adj[e1[1]][e1[0]] = 1
                            adj[e2[0]][e2[1]] = 1
                            adj[e2[1]][e2[0]] = 1
                        
                        if improved:
                            break
            
            if time.time() > deadline:
                break
        
        return adj
    
    # Base: Grötzsch graph on vertices 0-10
    base_adj = make_adj(groetzsch_edges)
    
    # Find maximum independent sets in Grötzsch graph
    def find_independent_sets(adj, vertices, min_size=4):
        results = []
        
        def backtrack(remaining, current_set):
            if len(current_set) >= min_size:
                results.append(current_set[:])
            
            if not remaining:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Include v
            new_remaining = [u for u in rest if not adj[v][u]]
            backtrack(new_remaining, current_set + [v])
            
            # Exclude v
            backtrack(rest, current_set)
        
        backtrack(list(vertices), [])
        return results
    
    grotzsch_vertices = list(range(11))
    isets = find_independent_sets(base_adj, grotzsch_vertices, min_size=4)
    
    # Try different configurations
    configs = []
    
    # Config 1: Connect new vertices to maximum independent sets
    for iset in isets:
        if len(iset) < 4:
            continue
        config = [row[:] for row in base_adj]
        for nv in range(11, 16):
            for iv in iset:
                config[nv][iv] = 1
                config[iv][nv] = 1
        configs.append(config)
    
    # Config 2: Try different independent sets of size 3
    isets3 = find_independent_sets(base_adj, grotzsch_vertices, min_size=3)
    for iset in isets3[:20]:  # Limit to avoid too many configs
        if len(iset) < 3:
            continue
        config = [row[:] for row in base_adj]
        for nv in range(11, 16):
            for iv in iset:
                config[nv][iv] = 1
                config[iv][nv] = 1
        configs.append(config)
    
    # Config 3: Random connections
    for _ in range(10):
        config = [row[:] for row in base_adj]
        for nv in range(11, 16):
            # Connect to random subset of Grötzsch vertices
            for gv in range(11):
                if random.random() < 0.5:
                    config[nv][gv] = 1
                    config[gv][nv] = 1
        # Check triangle-free
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if config[i][j] and config[j][k] and config[k][i]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            configs.append(config)
    
    # Evaluate and improve each config
    for config in configs:
        if time.time() > deadline:
            break
        
        # Check if triangle-free
        valid_tf = True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if config[i][j] and config[j][k] and config[k][i]:
                        valid_tf = False
                        break
                if not valid_tf:
                    break
            if not valid_tf:
                break
        
        if not valid_tf:
            continue
        
        # Check if not 3-colorable
        if is_3_colorable(config):
            continue
        
        # This is a valid starting point
        m = count_edges(config)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if config[i][j]]
        
        # Local search
        improved_adj = local_search(config, max_iter=2000)
        m = count_edges(improved_adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if improved_adj[i][j]]
    
    # If no valid config found, try random restarts
    if best_edges is None:
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random triangle-free graph
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        if is_triangle_free_add(adj, i, j):
                            adj[i][j] = 1
                            adj[j][i] = 1
            
            if is_3_colorable(adj):
                continue
            
            m = count_edges(adj)
            if m > best_score:
                best_score = m
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            
            improved_adj = local_search(adj, max_iter=1000)
            m = count_edges(improved_adj)
            if m > best_score:
                best_score = m
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if improved_adj[i][j]]
    
    # Fallback
    if best_edges is None:
        best_edges = [[u, v] for u, v in groetzsch_edges]
    
    return best_edges
