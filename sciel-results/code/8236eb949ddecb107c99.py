import time
import random

N = 20
C = 20
DEADLINE = 145.0
START = time.time()

def elapsed():
    return time.time() - START

def construct():
    best_edges = None
    best_count = 0
    
    # Use a faster approach: maintain adjacency and forbidden colour masks
    # adj[u][v] = colour or -1
    # For each pair (u,v), forbidden[u][v] is a bitmask of colours that would create a non-rainbow 4-cycle
    
    def try_full_construction(seed_val):
        random.seed(seed_val)
        adj = [[-1] * N for _ in range(N)]
        col_edges = []
        
        # Precompute all pairs
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        # Order pairs by some heuristic - try to add edges in an order that's likely to work
        # Sort by sum of degrees (will be 0 initially, so just randomize)
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                break
            if adj[u][v] >= 0:
                continue
            
            # Find which colours are safe for (u, v)
            # A colour c is forbidden if there exists x, y such that:
            # u-x, x-y, y-v all exist, and {c, adj[u][x], adj[x][y], adj[y][v]} has < 4 distinct colours
            # This happens iff c is in {adj[u][x], adj[x][y], adj[y][v]} AND at least two of those three are equal
            
            forbidden = [False] * C
            for x in range(N):
                if x == u or x == v or adj[u][x] < 0:
                    continue
                cu_x = adj[u][x]
                for y in range(N):
                    if y == u or y == v or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                        continue
                    cx_y = adj[x][y]
                    cy_v = adj[y][v]
                    # Check if any colour in {cu_x, cx_y, cy_v} that equals another would make c forbidden
                    # c is forbidden if c == cu_x and (cu_x == cx_y or cu_x == cy_v)
                    # or c == cx_y and (cx_y == cu_x or cx_y == cy_v)
                    # or c == cy_v and (cy_v == cu_x or cy_v == cx_y)
                    if cu_x == cx_y:
                        forbidden[cu_x] = True
                    if cu_x == cy_v:
                        forbidden[cu_x] = True
                    if cx_y == cy_v:
                        forbidden[cx_y] = True
            
            # Try colours in random order
            colours = list(range(C))
            random.shuffle(colours)
            
            placed = False
            for c in colours:
                if not forbidden[c]:
                    adj[u][v] = c
                    adj[v][u] = c
                    col_edges.append([u, v, c])
                    placed = True
                    break
            
            if not placed:
                continue
        
        return col_edges
    
    def local_search(col_edges):
        """Try to add more edges by reordering or trying different colour assignments."""
        adj = [[-1] * N for _ in range(N)]
        for u, v, c in col_edges:
            adj[u][v] = c
            adj[v][u] = c
        
        # Try adding missing edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_pairs)
        
        improved = True
        while improved and elapsed() < DEADLINE - 5:
            improved = False
            for u, v in all_pairs:
                if elapsed() > DEADLINE - 5:
                    break
                if adj[u][v] >= 0:
                    continue
                
                forbidden = [False] * C
                for x in range(N):
                    if x == u or x == v or adj[u][x] < 0:
                        continue
                    cu_x = adj[u][x]
                    for y in range(N):
                        if y == u or y == v or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                            continue
                        cx_y = adj[x][y]
                        cy_v = adj[y][v]
                        if cu_x == cx_y:
                            forbidden[cu_x] = True
                        if cu_x == cy_v:
                            forbidden[cu_x] = True
                        if cx_y == cy_v:
                            forbidden[cx_y] = True
                
                colours = list(range(C))
                random.shuffle(colours)
                for c in colours:
                    if not forbidden[c]:
                        adj[u][v] = c
                        adj[v][u] = c
                        col_edges.append([u, v, c])
                        improved = True
                        break
        
        return col_edges
    
    def verify(col_edges):
        """Verify the coloring is valid."""
        adj = [[-1] * N for _ in range(N)]
        for u, v, c in col_edges:
            adj[u][v] = c
            adj[v][u] = c
        
        for a in range(N):
            for b in range(a + 1, N):
                for c_idx in range(b + 1, N):
                    for d in range(c_idx + 1, N):
                        for (x, y, z, w) in [(a, b, c_idx, d), (a, b, d, c_idx), (a, c_idx, b, d)]:
                            if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                                cols = {adj[x][y], adj[y][z], adj[z][w], adj[w][x]}
                                if len(cols) != 4:
                                    return False
        return True
    
    # Try multiple seeds with greedy construction
    max_attempts = 50
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 10:
            break
        
        result = try_full_construction(i)
        count = len(result)
        
        if count > best_count:
            best_edges = result
            best_count = count
            
            # Try local improvement
            if elapsed() < DEADLINE - 15 and count > 0:
                improved = local_search([e[:] for e in best_edges])
                improved_count = len(improved)
                if improved_count > best_count:
                    best_edges = improved
                    best_count = improved_count
    
    # Final verification
    if best_edges and verify(best_edges):
        return best_edges
    
    # Fallback: star graph
    star = [[0, v, v] for v in range(1, N)]
    return star
