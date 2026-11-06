"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Try n=17 with distance-based coloring
    n = 17
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = min(j - i, n - (j - i))
            colors.append((d - 1) % 4)
    
    # Verify: check each color class for C6
    valid = True
    for c in range(4):
        if time.time() - start_time > 100:
            valid = False
            break
        edges = set()
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == c:
                    edges.add((i, j))
                idx += 1
        if _has_c6(edges, n):
            valid = False
            break
    
    if valid and time.time() - start_time < 110:
        return {'n': n, 'colors': colors}
    
    # Fallback: n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _has_c6(edges, n):
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    for verts in itertools.combinations(range(n), 6):
        v0 = verts[0]
        rest = list(verts[1:])
        for perm in itertools.permutations(rest):
            v1,v2,v3,v4,v5 = perm
            if v1 > v5:
                continue
            if (adj[v0][v1] and adj[v1][v2] and 
                adj[v2][v3] and adj[v3][v4] and 
                adj[v4][v5] and adj[v5][v0]):
                return True
    return False
# EVOLVE-BLOCK-END
