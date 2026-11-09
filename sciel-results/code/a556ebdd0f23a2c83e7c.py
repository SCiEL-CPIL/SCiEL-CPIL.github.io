import random
import time

def has_c6_in_color(edge_set, n):
    """Check if the graph with edges in edge_set contains a C6."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for C6
    # A C6 is a cycle of length 6.
    # We can iterate over all possible 6-tuples (a, b, c, d, e, f) with a < b < c < d < e < f
    # and check if (a,b), (b,c), (c,d), (d,e), (e,f), (f,a) are all in edge_set.
    # This is O(n^6) which is too slow for n=17.
    
    # Better: use DFS to find cycles of length 6.
    # Or use a simpler check: for each edge (u,v), check if there is a path of length 4 from u to v that doesn't use u or v.
    
    # Let's use a simpler heuristic: if the number of edges is large, it's likely to have a C6.
    # But we need an exact check.
    
    # For n=17, n^6 is 24 million, which might be too slow in Python.
    # We can optimize by only checking cycles that include a specific edge.
    
    for u, v in edge_set:
        # Find all paths of length 4 from u to v that don't use u or v
        # Path: u - a - b - c - d - v
        # a, b, c, d are distinct and not u or v
        for a in adj[u]:
            if a == v: continue
            for b in adj[a]:
                if b == u or b == a or b == v: continue
                for c in adj[b]:
                    if c == a or c == b or c == u or c == v: continue
                    for d in adj[c]:
                        if d == b or d == c or d == u or d == v: continue
                        if (d, v) in edge_set or (v, d) in edge_set:
                            return True
    return False

def is_valid(colors, n):
    color_edges = [[] for _ in range(4)]
    for i in range(n):
        for j in range(i+1, n):
            idx = i*n - i*(i+1)//2 + j - i - 1
            color_edges[colors[idx]].append((i, j))
    
    for c in range(4):
        if has_c6_in_color(set(color_edges[c]), n):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # Try to find a valid coloring for n=17
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for n in range(10, 18):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
            if is_valid(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        if found:
            # Try to extend to larger n
            continue
        else:
            break
    
    return {'n': best_n, 'colors': best_colors}
