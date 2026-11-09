def construct():
    import random
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 115

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]

    def is_valid(adj, neighbors):
        """Check if the current coloring is valid (all 4-cycles are rainbow)."""
        for u in range(n):
            for i, v in enumerate(neighbors[u]):
                for x in neighbors[v]:
                    if x == u:
                        continue
                    for w in neighbors[u]:
                        if w == v or w == x:
                            continue
                        if adj[w][x] != -1:
                            c1 = adj[u][v]
                            c2 = adj[v][x]
                            c3 = adj[x][w]
                            c4 = adj[w][u]
                            if len({c1, c2, c3, c4}) != 4:
                                return False
        return True

    def check_4cycle_conflict(adj, neighbors, u, v, c):
        """Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle."""
        for w in neighbors[u]:
            for x in neighbors[v]:
                if w != x and adj[w][x] != -1:
                    c1 = adj[u][w]
                    c2 = adj[w][x]
                    c3 = adj[x][v]
                    # The 4-cycle is u-v-x-w-u with colors c, c3, c2, c1
                    if len({c, c1, c2, c3}) != 4:
                        return True
        return False

    def try_random_construction():
        """Try one random construction."""
        random.shuffle(all_edges)
        adj = [[-1]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            valid_color = -1
            colors = list(range(20))
            random.shuffle(colors)
            
            for c in colors:
                if not check_4cycle_conflict(adj, neighbors, u, v, c):
                    valid_color = c
                    break
            
            if valid_color != -1:
                adj[u][v] = valid_color
                adj[v][u] = valid_color
                neighbors[u].append(v)
                neighbors[v].append(u)
                current_edges.append([u, v, valid_color])
        
        return current_edges, adj, neighbors

    def try_improve(edges, adj, neighbors):
        """Try to add more edges by removing some and re-adding."""
        best_edges = edges[:]
        best_adj = [row[:] for row in adj]
        best_neighbors = [lst[:] for lst in neighbors]
        
        # Try removing each edge and see if we can add more
        for i in range(len(edges)):
            u, v, c = edges[i]
            # Temporarily remove
            best_adj[u][v] = -1
            best_adj[v][u] = -1
            best_neighbors[u].remove(v)
            best_neighbors[v].remove(u)
            remaining = edges[:i] + edges[i+1:]
            
            # Try to add back all edges including the removed one
            new_edges = remaining[:]
            new_adj = [row[:] for row in best_adj]
            new_neighbors = [lst[:] for lst in best_neighbors]
            
            # Try to add the removed edge back with possibly different color
            for c2 in range(20):
                if not check_4cycle_conflict(new_adj, new_neighbors, u, v, c2):
                    new_adj[u][v] = c2
                    new_adj[v][u] = c2
                    new_neighbors[u].append(v)
                    new_neighbors[v].append(u)
                    new_edges.append([u, v, c2])
                    break
            
            # Try to add other missing edges
            for (a, b) in all_edges:
                if new_adj[a][b] == -1:
                    for c3 in range(20):
                        if not check_4cycle_conflict(new_adj, new_neighbors, a, b, c3):
                            new_adj[a][b] = c3
                            new_adj[b][a] = c3
                            new_neighbors[a].append(b)
                            new_neighbors[b].append(a)
                            new_edges.append([a, b, c3])
                            break
            
            if len(new_edges) > len(best_edges):
                best_edges = new_edges
                best_adj = new_adj
                best_neighbors = new_neighbors
            else:
                # Restore original
                best_adj[u][v] = c
                best_adj[v][u] = c
                best_neighbors[u].append(v)
                best_neighbors[v].append(u)
                break
        
        return best_edges

    best_edges = []
    
    while time.time() < deadline:
        edges, adj, neighbors = try_random_construction()
        
        if len(edges) > len(best_edges):
            best_edges = edges[:]
        
        # Try local improvement if we have time
        if time.time() < deadline - 30 and len(edges) >= len(best_edges) - 5:
            improved = try_improve(edges, adj, neighbors)
            if len(improved) > len(best_edges):
                best_edges = improved[:]
    
    return best_edges
