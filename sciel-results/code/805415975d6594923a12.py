import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Helper: check if a graph is valid
    # Optimized: Only check pairs (s, t) with deg(s) == deg(t)
    def is_valid(adj, deg):
        # Group vertices by degree
        deg_map = {}
        for i in range(n):
            d = deg[i]
            if d not in deg_map:
                deg_map[d] = []
            deg_map[d].append(i)
        
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            # For each pair in this degree group, check for 3-path
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s = verts[i]
                    t = verts[j]
                    # Check if there is a path s-u-v-t
                    # u in N(s), v in N(t), edge (u, v)
                    # u != t, v != s, u != v
                    found = False
                    # Iterate neighbors of s
                    for u in range(n):
                        if u == s or u == t or not adj[s][u]:
                            continue
                        # Iterate neighbors of t
                        for v in range(n):
                            if v == s or v == t or v == u or not adj[t][v]:
                                continue
                            if adj[u][v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Start with K_{7,8} as baseline
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    best_adj = [[False] * n for _ in range(n)]
    for u in part_a:
        for v in part_b:
            best_adj[u][v] = True
            best_adj[v][u] = True
    
    best_deg = get_degrees(best_adj)
    best_edges = count_edges(best_adj)
    best_valid = is_valid(best_adj, best_deg)
    
    # If baseline is invalid, we need to find a valid one quickly
    # K_{7,8} is invalid.
    # Let's try a random graph with ~40 edges as a fallback if greedy fails to fix K_{7,8}
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Greedy Add
    for _ in range(100):  # Reduced passes due to faster check, but more iterations possible
        if time.time() > deadline:
            break
        improved = False
        random.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if best_adj[u][v]:
                continue
            # Try adding edge (u, v)
            best_adj[u][v] = True
            best_adj[v][u] = True
            new_deg = get_degrees(best_adj)
            if is_valid(best_adj, new_deg):
                best_deg = new_deg
                best_edges = count_edges(best_adj)
                best_valid = True
                improved = True
            else:
                # Revert
                best_adj[u][v] = False
                best_adj[v][u] = False
    
    # Local Search: Swap edges
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        random.shuffle(all_pairs)
        for i, (u1, v1) in enumerate(all_pairs):
            if time.time() > deadline:
                break
            if not best_adj[u1][v1]:
                continue
            for (u2, v2) in all_pairs:
                if time.time() > deadline:
                    break
                if best_adj[u2][v2] or (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                # Try swap: remove (u1,v1), add (u2,v2)
                best_adj[u1][v1] = False
                best_adj[v1][u1] = False
                best_adj[u2][v2] = True
                best_adj[v2][u2] = True
                new_deg = get_degrees(best_adj)
                new_edges = count_edges(best_adj)
                if is_valid(best_adj, new_deg):
                    # Accept if valid, prefer more edges (though swap keeps count same)
                    # Actually swap keeps edge count same. We only care about validity here to enable future adds.
                    best_deg = new_deg
                    best_edges = new_edges
                    best_valid = True
                    improved = True
                else:
                    # Revert
                    best_adj[u1][v1] = True
                    best_adj[v1][u1] = True
                    best_adj[u2][v2] = False
                    best_adj[v2][u2] = False
        if not improved:
            break
            
    # If we didn't find a valid graph, try a random restart with a sparser graph
    if not best_valid:
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            adj = [[False] * n for _ in range(n)]
            # Random graph with p=0.4
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.4:
                        adj[u][v] = True
                        adj[v][u] = True
            deg = get_degrees(adj)
            if is_valid(adj, deg):
                best_adj = adj
                best_deg = deg
                best_edges = count_edges(adj)
                best_valid = True
                break
    
    # Build result
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
