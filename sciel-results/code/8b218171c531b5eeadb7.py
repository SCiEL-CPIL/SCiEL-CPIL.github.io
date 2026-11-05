import time
import random

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    n = 20
    num_colors = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # We will run multiple restarts with different random seeds
    # to explore the solution space.
    attempts = 0
    
    while _time_left() > 5.0:
        rng = random.Random(attempts)
        
        # Shuffle edges to change the greedy order
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        # adj[u][v] = color (0..19) or -1 if no edge
        adj = [[-1] * n for _ in range(n)]
        # neighbors[u] = list of v where adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in shuffled_edges:
            # Try to find a color for edge (u, v)
            placed = False
            # Randomize color order to avoid bias towards low colors
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for c in colors:
                # Check if adding (u, v, c) creates a non-rainbow 4-cycle
                # A 4-cycle is u-a-b-v-u.
                # Edges: (u,a), (a,b), (b,v), (v,u)
                # Colors: ca, cb, cv, c
                # Condition: len({ca, cb, cv, c}) == 4
                
                ok = True
                # Optimization: Iterate over neighbors of u
                for a in neighbors[u]:
                    if a == v:
                        continue
                    ca = adj[u][a]
                    # If ca is -1, it's not an edge, but neighbors[u] only contains edges.
                    # So ca is always valid color.
                    
                    # Iterate over neighbors of v
                    for b in neighbors[v]:
                        if b == u or b == a:
                            continue
                        
                        # Check if (a, b) is an edge
                        if adj[a][b] != -1:
                            cb = adj[a][b]
                            cv = adj[b][v]
                            
                            # Check rainbow condition
                            # Colors are ca, cb, cv, c
                            # They must be all distinct.
                            if ca == c or cb == c or cv == c:
                                ok = False
                                break
                            if ca == cb or ca == cv or cb == cv:
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    # Place edge
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    placed = True
                    break
            
            # If no color works, we skip this edge.
            # In a denser graph, we might want to reconsider, but greedy is the base.
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        attempts += 1
        
    # Fallback if no edges were found (should not happen)
    if not best_edges:
        best_edges = [[0, 1, 0]]
        
    return best_edges
