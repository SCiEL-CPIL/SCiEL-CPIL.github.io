import random
import time
import sys

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    n = 20
    num_colors = 20
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def is_valid_c4(adj, u, v, c):
        """Check if adding edge (u,v) with color c creates any non-rainbow C4."""
        # A 4-cycle involving (u,v) must be u-a-b-v-u
        # where (u,a), (a,b), (b,v) are existing edges
        # We need to check all pairs (a,b) where a~u, b~v, and a~b
        
        # Get neighbors of u and v (excluding each other for the new edge context)
        # Actually, neighbors[u] contains all current neighbors of u
        # We iterate over a in neighbors[u], b in neighbors[v]
        # Such that a != v, b != u, a != b, and adj[a][b] != -1
        
        # Optimization: iterate over the smaller set first?
        # But we need pairs. Let's just do it.
        
        # To speed up, we can check if the set of "forbidden" colors is large.
        # But let's just check each potential C4.
        
        # For each a in neighbors[u]:
        #   For each b in neighbors[v]:
        #     if a == b: continue
        #     if adj[a][b] != -1:
        #        ca = adj[u][a]
        #        cb = adj[a][b]
        #        cv = adj[b][v]
        #        if len({ca, cb, cv, c}) < 4: return False
        
        # This is O(deg(u) * deg(v)) per edge insertion.
        # With n=20, max deg is 19. So max 361 checks per edge.
        # 190 edges * 361 * 20 colors = ~1.3M operations per greedy run.
        # This is fast enough for many restarts.
        
        nu = neighbors[u]
        nv = neighbors[v]
        
        # Iterate
        for a in nu:
            if a == v:
                continue
            ca = adj[u][a]
            for b in nv:
                if b == u or b == a:
                    continue
                cb = adj[a][b]
                if cb != -1:
                    cv = adj[b][v]
                    # Check distinctness
                    if ca == c or cb == c or cv == c or ca == cb or ca == cv or cb == cv:
                        return False
        return True

    def greedy_build(seed):
        rng = random.Random(seed)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in shuffled_edges:
            placed = False
            # Randomize color order
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for c in colors:
                if is_valid_c4(adj, u, v, c):
                    # Place edge
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    placed = True
                    break
            
            # If no color works, skip edge
        
        return edges

    def improve(edges, seed):
        """Try to add more edges or swap colors to allow more edges."""
        # This is a placeholder for a local search improvement.
        # For now, just return the input.
        return edges

    best_edges = None
    best_count = 0
    
    # Multiple restarts
    # Estimate time per greedy_build:
    # 190 edges, each tries up to 20 colors.
    # Each check is O(deg^2).
    # Let's run as many as possible.
    
    # First, run a few to get a baseline
    for i in range(100):
        if time.time() > deadline - 10:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Try to improve the best solution found so far?
    # Or just do more restarts with different random seeds.
    
    # Use a larger range of seeds
    for i in range(100, 10000):
        if time.time() > deadline - 5:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we hit a very high count, we might stop early?
            # Max possible is 190. But C4 constraint limits it.
            # If best_count >= 100, it's quite good.
            if best_count >= 120:
                # Keep searching for a bit more
                pass

    if best_edges is None:
        # Fallback: Star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges

