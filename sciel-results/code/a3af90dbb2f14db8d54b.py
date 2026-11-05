import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin for safety
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
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

    # Multiple restarts with different orderings
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 200 # Increased restarts possible due to speedup?
    # Let's keep 500 but check time. The speedup is significant.
    # Old: ~4500 checks/edge. New: ~225 checks/edge + small overhead. ~20x speedup.
    # So 500 restarts should be fast.
    
    for ot in order_types:
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            edges = greedy_build(i, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        # Fallback: Star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
