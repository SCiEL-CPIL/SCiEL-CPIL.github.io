import random
import time

def construct():
    deadline = time.time() + 130  # 20s margin for safety
    n = 20
    num_colors = 20
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def greedy_build(seed, order_type='random'):
        rng = random.Random(seed)
        edges_list = all_edges[:]
        
        if order_type == 'random':
            rng.shuffle(edges_list)
        elif order_type == 'natural':
            pass # Already sorted by u, then v
        elif order_type == 'reverse':
            edges_list.reverse()
            
        # adj[u][v] = color if edge exists, -1 otherwise
        adj = [[-1] * n for _ in range(n)]
        # neighbors[u] = set of v such that adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in edges_list:
            # Check if edge can be added
            # Find all length-3 paths u-a-b-v
            # If any path has non-distinct colors, edge is impossible.
            # Otherwise, forbidden colors are the colors on those paths.
            
            impossible = False
            forbidden_colors = set()
            
            # Iterate a in neighbors[u]
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                # Iterate b in neighbors[v]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    # Check if (a,b) is an edge
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        # Check distinctness of ca, cb, cv
                        if ca == cb or cb == cv or ca == cv:
                            impossible = True
                            break
                        # Add to forbidden
                        forbidden_colors.add(ca)
                        forbidden_colors.add(cb)
                        forbidden_colors.add(cv)
                if impossible:
                    break
            
            if not impossible:
                # Find a valid color
                # Try colors in random order to avoid bias? Or just 0..19?
                # Random order might help find a solution if one exists, but here we just need ANY valid color.
                # Since we just need to pick one, and we want to maximize edges,
                # the choice of color might affect future edges.
                # Randomizing color choice adds diversity.
                colors = list(range(num_colors))
                rng.shuffle(colors)
                
                placed = False
                for c in colors:
                    if c not in forbidden_colors:
                        # Place edge
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        placed = True
                        break
        
        return edges

    def local_search(edges, seed, max_iters=100):
        """
        Try to add back removed edges.
        edges: list of [u, v, c]
        """
        rng = random.Random(seed)
        n = 20
        num_colors = 20
        
        # Build state
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        # Candidate edges: all edges not in graph
        present = set((min(u,v), max(u,v)) for u, v, c in edges)
        candidates = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in present]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            rng.shuffle(candidates)
            for u, v in candidates:
                # Check if we can add edge (u, v)
                impossible = False
                forbidden_colors = set()
                for a in neighbors[u]:
                    if a == v:
                        continue
                    ca = adj[u][a]
                    for b in neighbors[v]:
                        if b == u or b == a:
                            continue
                        if adj[a][b] != -1:
                            cb = adj[a][b]
                            cv = adj[b][v]
                            if ca == cb or cb == cv or ca == cv:
                                impossible = True
                                break
                            forbidden_colors.add(ca)
                            forbidden_colors.add(cb)
                            forbidden_colors.add(cv)
                    if impossible:
                        break
                
                if not impossible:
                    # Try to find a color
                    # We can also try to change colors of existing edges? 
                    # For now, just try to add with any valid color.
                    # If no color available, we might need to recolor.
                    # But if forbidden_colors < 20, there is a free color.
                    # If forbidden_colors == 20, we are stuck unless we recolor.
                    # Let's just try to add if possible.
                    found_color = False
                    for c in range(num_colors):
                        if c not in forbidden_colors:
                            adj[u][v] = c
                            adj[v][u] = c
                            neighbors[u].add(v)
                            neighbors[v].add(u)
                            edges.append([u, v, c])
                            present.add((u, v))
                            candidates.remove((u, v))
                            improved = True
                            found_color = True
                            break
                    # If found_color is false, we couldn't add it.
        
        return edges

    best_edges = None
    best_count = 0
    
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 100 # Start with fewer, polish later
    
    for ot in order_types:
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            edges = greedy_build(i, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]

    # Polish the best solution found
    if best_edges is not None:
        # Run local search on the best edge set
        # Try a few seeds for local search
        for ls_seed in range(20):
            if time.time() > deadline - 2:
                break
            # Work on a copy
            current_edges = best_edges[:]
            polished = local_search(current_edges, ls_seed, max_iters=50)
            if len(polished) > best_count:
                best_count = len(polished)
                best_edges = polished

    if best_edges is None:
        # Fallback: Star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
