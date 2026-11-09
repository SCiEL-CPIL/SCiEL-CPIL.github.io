import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Try n=18, then 17, etc.
    for n in range(18, 0, -1):
        if time.time() > deadline:
            break
        # Run search for n
        result = search_coloring(n, deadline)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: return a known valid small coloring if all else fails
    # The seed program returns n=9. We should at least return that.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def search_coloring(n, deadline):
    total_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Pre-allocate adjacency lists for each color
    adj = [set() for _ in range(4)] # adj[c] is a set of (u,v) or list of lists?
    # Using sets of neighbors is faster for BFS/DFS
    neighbors = [[set() for _ in range(n)] for _ in range(4)]
    
    # Assignments array
    colors = [None] * total_edges
    
    # Symmetry breaking:
    # If n > 1, edge (0,1) is index 0. Fix to 0.
    # If n > 2, edge (0,2) is index 1. Fix to 1.
    # If n > 3, edge (0,3) is index 2. Fix to 2.
    
    def has_simple_path_len_5(c, u, v):
        # Check if there is a simple path of length 5 from u to v in graph c
        # DFS
        if u == v: return False
        visited = {u}
        stack = [(u, 0)]
        while stack:
            node, depth = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            for nei in neighbors[c][node]:
                if nei not in visited:
                    visited.add(nei)
                    stack.append((nei, depth + 1))
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if idx == total_edges:
            return colors[:]
        
        u, v = edges[idx]
        
        # Symmetry breaking for first few edges
        if idx == 0 and n > 1:
            candidates = [0]
        elif idx == 1 and n > 2:
            candidates = [1]
        elif idx == 2 and n > 3:
            candidates = [2]
        else:
            candidates = [0, 1, 2, 3]
            random.shuffle(candidates)
            
        for c in candidates:
            # Check if adding (u,v) to color c creates a C6
            # i.e., is there a simple path of length 5 between u and v in color c?
            if not has_simple_path_len_5(c, u, v):
                # Assign
                colors[idx] = c
                neighbors[c][u].add(v)
                neighbors[c][v].add(u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Unassign
                colors[idx] = None
                neighbors[c][u].remove(v)
                neighbors[c][v].remove(u)
        
        return None

    # Run a few randomized restarts
    # The symmetry breaking fixes the first 3 edges, so we don't need many restarts for those.
    # But the rest is randomized.
    # If the first run fails, we can try again with different random seeds or just let it backtrack.
    # Backtracking is exhaustive within the symmetry-broken space.
    # So one call to backtrack should be sufficient if it explores the whole space.
    # But if it's too slow, we might time out.
    # Given n=18, the space is huge.
    # We should probably limit the depth or use a heuristic.
    # However, for n=17, it might be solvable.
    
    # Let's try n=18 first with a strict time limit per attempt?
    # No, just let it run until deadline.
    
    # To handle the case where n=18 is too hard, we might want to try n=17 sooner.
    # But the task says "Maximize n".
    # So we must try 18.
    
    # One optimization: Order edges.
    # Current order is lexicographic.
    # This is generally good.
    
    return backtrack(0)
