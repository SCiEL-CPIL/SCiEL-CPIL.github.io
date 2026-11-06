import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0

    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        # Fast backtracking 3-coloring with bitmasks
        # We use a simple recursive approach but optimized with bit operations
        # for checking constraints.
        
        # To improve pruning, we can order vertices. However, for n=16,
        # fixed order 0..15 is often fine if the graph is dense.
        # Let's try to add a simple heuristic: if a vertex has no available colors, fail.
        
        color = [-1] * n
        
        # Precompute neighbor masks for quick access? adj is already bitmasks.
        
        def backtrack(i):
            if i == n:
                return True
            
            # Check constraints: no neighbor j < i can have the same color
            # We only need to check neighbors j < i because neighbors j > i
            # are not colored yet.
            
            # Get mask of neighbors with index < i
            mask_lower = adj[i] & ((1 << i) - 1)
            
            # Determine used colors by neighbors
            used_colors = 0
            m = mask_lower
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                c = color[j]
                if c != -1:
                    used_colors |= (1 << c)
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

    def greedy_expand(adj, shuffle=True):
        # Create a list of all possible edges not currently present
        possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    possible.append((u, v))
        
        if shuffle:
            random.shuffle(possible)
        else:
            # Sort by potential: e.g., sum of degrees? Or just fixed order.
            # Fixed order might be more deterministic and fast.
            # Let's sort by u then v for determinism if shuffle is False.
            possible.sort()
        
        added = 0
        for u, v in possible:
            if is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, steps=1000):
        # Try to add edges, and if stuck, try removing an edge and re-adding to escape local optima?
        # For now, just greedy add is strong.
        # A more advanced local search would try to swap edges to increase count while maintaining constraints.
        # But greedy add until no more edges can be added is a local maximum.
        # To escape, we could randomly remove an edge and try to re-add.
        
        # Let's implement a simple simulated annealing or random restart local search.
        # Since we have time, we can do multiple random restarts of greedy.
        
        # This function just does one greedy pass.
        return greedy_expand(adj, shuffle=True)

    # Strategy 1: Grötzsch seed
    # Strategy 2: Random seed (if 4-chromatic)
    
    # We will alternate between these.
    
    # To improve the score, we should try to find a graph with more edges.
    # The Grötzsch graph has 11 vertices and 20 edges.
    # With 16 vertices, we can add 5 more vertices.
    # The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4) = 64 for n=16.
    # But it must be 4-chromatic.
    
    # Let's run the loop.
    
    while time.time() < deadline:
        # Choose a starting point
        if random.random() < 0.6:
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
                    if random.random() < 0.4:
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
            else:
                # Try to make it 4-chromatic by adding edges?
                # We can try to add edges that force a 4th color.
                # This is hard to do greedily.
                # Maybe try to add a Grötzsch subgraph?
                pass

    return best_edges
