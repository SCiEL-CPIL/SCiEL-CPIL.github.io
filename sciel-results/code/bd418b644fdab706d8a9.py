import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Seed solution (n=9 from the original)
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best_result = {'n': seed_n, 'colors': seed_colors}
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 15:
            break
            
        # Time budget
        budget = 120.0 if n == 17 else 10.0
        target_end = time.time() + budget
        
        res = _backtrack_search(n, target_end)
        if res:
            return res
            
    return best_result

def _backtrack_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge index
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    # For C6 detection, we need to check if adding an edge (u,v) of color c
    # creates a C6 in color c. This happens if there is a path of length 5
    # from u to v in the graph of color c, using only previously assigned edges.
    
    # We'll use a recursive backtracking with pruning.
    # State: colors assigned so far, adjacency lists per color.
    
    colors = [-1] * num_edges
    adj = [[] for _ in range(4)]  # adj[c] is list of lists
    
    # For efficiency, we'll maintain the adjacency as sets for quick lookup
    adj_sets = [set() for _ in range(4)]
    # adj_sets[c] is a set of frozensets? No, let's use a dict of sets
    # adj_sets[c][u] = set of v such that (u,v) is color c
    
    adj_dict = [dict() for _ in range(4)]
    for c in range(4):
        for u in range(n):
            adj_dict[c][u] = set()
    
    def has_c6_in_color(c, u, v):
        """Check if there's a path of length 5 from u to v in color c,
        not using edge (u,v)."""
        # BFS/DFS from u, depth 5, target v
        # Path: u -> x1 -> x2 -> x3 -> x4 -> v
        # All distinct, and (u,v) not used in the path (which is automatic since we stop at v at depth 5)
        
        # We'll do DFS with visited set
        # To speed up, we can use the fact that we're adding edges one by one,
        # so the graph is sparse initially.
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            for nei in adj_dict[c][node]:
                if nei in visited:
                    continue
                if nei == v and depth < 4:
                    continue
                if nei == u and depth < 4:
                    continue
                visited.add(nei)
                if dfs(nei, depth + 1, visited):
                    return True
                visited.remove(nei)
            return False
        
        return dfs(u, 0, {u})
    
    def is_valid_addition(c, u, v):
        """Check if adding edge (u,v) of color c creates a C6."""
        # A C6 is formed if there's a path of length 5 from u to v in color c
        # that doesn't use edge (u,v).
        # But wait: the path of length 5 from u to v, combined with edge (u,v),
        # forms a cycle of length 6. The vertices in the path must be distinct
        # and not include u or v (except at the ends).
        # So we need a path u -> x1 -> x2 -> x3 -> x4 -> v where
        # x1,x2,x3,x4 are distinct and not u or v.
        
        # The DFS above handles this with the visited set and the checks.
        return has_c6_in_color(c, u, v)
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        for c in range(4):
            # Check if adding (u,v) with color c is valid
            if is_valid_addition(c, u, v):
                continue
                
            # Try color c
            colors[idx] = c
            adj_dict[c][u].add(v)
            adj_dict[c][v].add(u)
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[idx] = -1
            adj_dict[c][u].remove(v)
            adj_dict[c][v].remove(u)
        
        return False
    
    # To make the search more effective, we can use a heuristic ordering of edges
    # and random restarts.
    
    # Let's try multiple random restarts with different edge orderings
    best_colors = None
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Shuffle edge order
        shuffled_idx = list(range(num_edges))
        random.shuffle(shuffled_idx)
        
        # Reset state
        colors = [-1] * num_edges
        adj_dict = [dict() for _ in range(4)]
        for c in range(4):
            for u in range(n):
                adj_dict[c][u] = set()
        
        # Use a recursive function with the shuffled order
        def backtrack_shuffled(pos):
            if time.time() > deadline:
                return False
            
            if pos == num_edges:
                return True
            
            idx = shuffled_idx[pos]
            u, v = edges[idx]
            
            # Try colors in a somewhat random order
            color_order = list(range(4))
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid_addition(c, u, v):
                    continue
                    
                colors[idx] = c
                adj_dict[c][u].add(v)
                adj_dict[c][v].add(u)
                
                if backtrack_shuffled(pos + 1):
                    return True
                
                colors[idx] = -1
                adj_dict[c][u].remove(v)
                adj_dict[c][v].remove(u)
            
            return False
        
        if backtrack_shuffled(0):
            best_colors = colors[:]
            break
    
    if best_colors is not None:
        return {'n': n, 'colors': best_colors}
    
    return None

