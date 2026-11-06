"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 40
    
    def is_c6_free(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency matrix adj creates a C6."""
        if adj[u][v]:
            return True
        
        # Neighbors of u and v
        nu = [x for x in range(n) if adj[u][x] and x != v]
        nv = [x for x in range(n) if adj[v][x] and x != u]
        
        # For each a in nu, d in nv, check if path a-b-c-d exists in G \ {u,v}
        # a-b-c-d length 3
        for a in nu:
            for d in nv:
                if a == d:
                    continue
                # Check path a-b-c-d
                # b must be neighbor of a, not u,v,a,d
                # c must be neighbor of b and d, not u,v,a,b,d
                for b in range(n):
                    if b in (u, v, a, d):
                        continue
                    if not adj[a][b]:
                        continue
                    # Check if there is a c
                    # c in N(b) ∩ N(d) \ {u, v, a, b, d}
                    for c in range(n):
                        if c in (u, v, a, b, d):
                            continue
                        if adj[b][c] and adj[c][d]:
                            return False
        return True

    def count_edges(adj):
        return sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])

    random.seed(42)
    best_edges = []
    best_count = 0
    
    # Use multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        
        # Random initialization: pick random edges that don't form C6
        # Start with empty graph, add random edges
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        added = 0
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if is_c6_free(adj, u, v):
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
                added += 1
        
        # Local search: try to swap edges to increase count?
        # Or just greedy addition is often enough if we restart.
        # Let's try a simple local search: try to add any missing edge.
        # This is effectively what the loop above does if we iterate until no more can be added.
        # But the loop above only adds if it was in the shuffled list.
        # Let's do a second pass to catch any missed edges.
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not adj[u][v]:
                if is_c6_free(adj, u, v):
                    adj[u][v] = adj[v][u] = True
                    edges.append([u, v])
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edges)
            
        if best_count >= 105:
            break
            
    return best_edges
# EVOLVE-BLOCK-END
