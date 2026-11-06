import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try n=18
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        # Allocate time based on n
        if n == 18:
            time_limit = 100
        elif n == 17:
            time_limit = 20
        elif n == 16:
            time_limit = 10
        else:
            time_limit = 5
            
        if time.time() > deadline - time_limit:
            continue
            
        result = search_coloring(n, time_limit)
        if result is not None:
            return result
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def search_coloring(n, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    # adj[color][u][v]
    adj = [[[0]*n for _ in range(n)] for _ in range(4)]
    colors = [0] * (n*(n-1)//2)
    edge_idx = 0
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def has_c6_simple_path(u, v, color, depth, visited):
        # Check if there is a simple path of length 'depth' from u to v in adj[color]
        # We are adding edge (u,v), so we must not use it.
        # But adj[color][u][v] is currently 0 in the check context if we check before setting?
        # Better: check on the graph WITHOUT the new edge.
        # So we just traverse adj[color] where adj[color][u][v] is 0 (since we haven't set it yet).
        
        if depth == 0:
            return u == v
        if depth == 1:
            return adj[color][u][v] == 1
        
        # Optimization: if depth is large, prune
        for w in range(n):
            if w == u or w == v:
                continue
            if w not in visited and adj[color][u][w] == 1:
                visited.add(w)
                if has_c6_simple_path(w, v, color, depth-1, visited):
                    visited.remove(w)
                    return True
                visited.remove(w)
        return False

    def is_valid_add(u, v, color):
        # Check if adding (u,v) with 'color' creates a C6.
        # This happens if there is a simple path of length 5 from u to v in the current graph of 'color'.
        # Current graph does not include (u,v).
        visited = {u, v}
        # We need path u -> ... -> v of length 5.
        # Start from neighbors of u.
        # Path: u -> x1 -> x2 -> x3 -> x4 -> v
        # x1 is neighbor of u.
        # We need path from x1 to v of length 4 in G \ {u}.
        # Actually, simpler:
        # Just call has_c6_simple_path(u, v, color, 5, {u})
        # But we must ensure the path doesn't use v until the end?
        # The function above allows visiting v only if depth matches.
        # Let's refine:
        # A simple path of length 5 from u to v.
        # Vertices u, x1, x2, x3, x4, v are distinct.
        
        def dfs(cur, target, depth, visited):
            if depth == 0:
                return cur == target
            if depth == 1:
                return adj[color][cur][target] == 1 and target not in visited
            for nxt in range(n):
                if nxt in visited:
                    continue
                if nxt == target and depth > 1:
                    continue # Can only reach target at last step
                if adj[color][cur][nxt] == 1:
                    visited.add(nxt)
                    if dfs(nxt, target, depth-1, visited):
                        visited.remove(nxt)
                        return True
                    visited.remove(nxt)
            return False

        visited = {u}
        return dfs(u, v, 5, visited)

    def backtrack(idx):
        nonlocal edge_idx
        if time.time() > deadline:
            return False
        
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        
        # Symmetry breaking for first few edges
        if idx == 0:
            # (0,1) must be 0
            if assign(0, 1, 0, idx):
                if backtrack(idx+1):
                    return True
            return False
        elif idx == 1:
            # (0,2) can be 0 or 1 (if 0, same as 0,1; if 1, new color)
            # If we pick 2, we can swap 1 and 2.
            for c in [0, 1]:
                if assign(0, 2, c, idx):
                    if backtrack(idx+1):
                        return True
            return False
        elif idx == 2:
            # (0,3)
            # Colors used so far: 0, and maybe 1.
            # If 1 not used, c in [0, 1].
            # If 1 used, c in [0, 1, 2]?
            # Let's just use [0, 1, 2, 3] but with a small heuristic.
            # Actually, just iterate 0..3.
            pass

        for c in range(4):
            if is_valid_add(u, v, c):
                if assign(u, v, c, idx):
                    if backtrack(idx+1):
                        return True
        return False

    def assign(u, v, c, idx):
        colors[idx] = c
        adj[c][u][v] = 1
        adj[c][v][u] = 1
        return True

    def unassign(u, v, c, idx):
        adj[c][u][v] = 0
        adj[c][v][u] = 0
        colors[idx] = 0

    # Wrapper for backtrack to handle unassign
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == len(edges):
            return True
        u, v = edges[idx]
        
        if idx == 0:
            cs = [0]
        elif idx == 1:
            cs = [0, 1]
        else:
            cs = [0, 1, 2, 3]
            
        for c in cs:
            if is_valid_add(u, v, c):
                assign(u, v, c, idx)
                if bt(idx+1):
                    return True
                unassign(u, v, c, idx)
        return False

    if bt(0):
        return {'n': n, 'colors': colors}
    return None
