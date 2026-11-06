import time
import random

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        # Backtracking 3-coloring with bitmasks for speed
        color = [-1] * n
        
        # Precompute neighbor masks for faster access? 
        # adj is already a list of bitmasks.
        
        def backtrack(i):
            if i == n:
                return True
            
            # Determine available colors based on neighbors with index < i
            # We can optimize by only checking neighbors j < i
            mask = adj[i]
            # Only consider neighbors j < i
            mask_lower = mask & ((1 << i) - 1)
            
            used_colors = 0
            # Iterate through set bits in mask_lower
            m = mask_lower
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if color[j] != -1:
                    used_colors |= (1 << color[j])
                m ^= lsb
            
            # Try colors 0, 1, 2
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[i] = c
                    if backtrack(i + 1):
                        return True
            color[i] = -1
            return False
        
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            # Only consider v > u
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def make_grotzsch_adj():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4
        for u, v in cycle:
            # (u, v') => (u, 5+v)
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            # (v, u') => (v, 5+u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def greedy_expand(adj):
        # Create a list of all possible edges not currently present
        possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    possible.append((u, v))
        
        random.shuffle(possible)
        
        added = 0
        for u, v in possible:
            if is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    # Strategy 1: Grötzsch seed
    # Strategy 2: Random seed (if 4-chromatic)
    
    while time.time() < deadline:
        if random.random() < 0.7:
            # Use Grötzsch seed
            adj = make_grotzsch_adj()
            # Greedy expand
            greedy_expand(adj)
            # It is guaranteed to be 4-chromatic because it contains Grötzsch
            edges = get_edges_from_adj(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
        else:
            # Random seed
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        if is_triangle_free_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            
            # Check if 4-chromatic
            if not is_3_colorable(adj):
                greedy_expand(adj)
                edges = get_edges_from_adj(adj)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges

    return best_edges
