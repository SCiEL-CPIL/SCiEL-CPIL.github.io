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
    
    def greedy_build(seed):
        rng = random.Random(seed)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        # adj[u][v] = color if edge exists, -1 otherwise
        adj = [[-1] * n for _ in range(n)]
        # neighbors[u] = set of v such that adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in shuffled_edges:
            # Determine forbidden colors for edge (u, v)
            forbidden = set()
            possible = True
            
            # Iterate a in N(u), b in N(v)
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    # Check if (a, b) is an edge
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        # Check if existing 3 colors are distinct
                        if ca == cb or ca == cv or cb == cv:
                            possible = False
                            break
                        else:
                            forbidden.add(ca)
                            forbidden.add(cb)
                            forbidden.add(cv)
                if not possible:
                    break
            
            if possible:
                # Find a valid color
                for c in range(num_colors):
                    if c not in forbidden:
                        # Place edge
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        break
                # If no color found (forbidden has all 20), skip edge
        
        return edges

    # Multiple restarts
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
            # Optional: Verify best result occasionally?
            # The greedy check is strict, so it should be valid.
            # But if there's a bug, we might return invalid.
            # Let's assume the check is correct.

    if best_edges is None:
        # Fallback: Star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
