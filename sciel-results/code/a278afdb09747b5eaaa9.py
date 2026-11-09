import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def has_c4_conflict(adj, u, v, c):
        """
        Check if adding edge (u,v) with color c creates a C4 in color c.
        adj is a list of lists, adj[c][i] is set of neighbors of i in color c.
        """
        neighbors_v = adj[c][v]
        neighbors_u = adj[c][u]
        
        # We need w in neighbors_v and x in neighbors_u such that (w,x) is in adj[c][w]
        # Cycle: u - v - w - x - u
        # Edges: (u,v) [new], (v,w), (w,x), (x,u)
        # (v,w) is in adj[c][v]
        # (x,u) is in adj[c][u]
        # (w,x) must be in adj[c][w]
        
        # Optimize: iterate over the smaller set
        if len(neighbors_v) > len(neighbors_u):
            neighbors_v, neighbors_u = neighbors_u, neighbors_v
            # Note: The logic is symmetric for existence of w,x
            # w in neighbors_v (original v), x in neighbors_u (original u)
            # We just need ANY pair w,x satisfying the condition.
            
        for w in neighbors_v:
            # w is neighbor of v
            # Check all x in neighbors_u
            for x in neighbors_u:
                # x is neighbor of u
                # Check if w and x are connected in color c
                if x in adj[c][w]:
                    return True
        return False

    def try_construct(n, max_restarts=1000):
        """
        Try to construct a valid 5-coloring for K_n.
        Returns list of colors or None.
        """
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize adjacency structures
            # adj[c][i] = set of neighbors
            adj = [[set() for _ in range(n)] for _ in range(5)]
            
            # Edges to process
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            success = True
            for u, v in edges:
                # Try colors in random order
                colors = [0, 1, 2, 3, 4]
                random.shuffle(colors)
                
                placed = False
                for c in colors:
                    if not has_c4_conflict(adj, u, v, c):
                        # Place edge
                        adj[c][u].add(v)
                        adj[c][v].add(u)
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                # Verify
                colors_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        # Find color
                        for c in range(5):
                            if j in adj[c][i]:
                                colors_list.append(c)
                                break
                return colors_list
        
        return None

    # Try increasing n
    # Known max is 26. Try 26, then 27, 28.
    # Start with 26 as it's more likely to succeed quickly.
    best_n = 1
    best_colors = None
    
    # We can start from a known good smaller n to guide the search?
    # For now, pure random restarts.
    
    for n in [26, 27, 28, 25, 24]:
        if time.time() > deadline - 5:
            break
        
        # Increase restarts for smaller n if needed, but 26 is the target.
        # For n=26, we might need many restarts.
        # Let's allocate time proportional to n.
        # If we fail 26 quickly, try 27? No, 27 is harder.
        # Try 26 first.
        
        colors = try_construct(n, max_restarts=2000)
        
        if colors is not None:
            best_n = n
            best_colors = colors
            # If we found 26, we can try 27 with remaining time
            if n == 26:
                # Try 27 with remaining time
                if time.time() < deadline - 10:
                    colors_27 = try_construct(27, max_restarts=500)
                    if colors_27 is not None:
                        best_n = 27
                        best_colors = colors_27
            # If we found 27, try 28?
            elif n == 27:
                if time.time() < deadline - 5:
                    colors_28 = try_construct(28, max_restarts=100)
                    if colors_28 is not None:
                        best_n = 28
                        best_colors = colors_28
            break # Found a valid one, move to next (larger) or stop if max reached
        
    # If no valid coloring found for n>=24, fallback to n=16 algebraic
    if best_colors is None:
        # Implement the n=16 GF(4) construction as fallback
        n = 16
        gf4_mult = {
            (0,0):0, (0,1):0, (0,2):0, (0,3):0,
            (1,0):0, (1,1):1, (1,2):2, (1,3):3,
            (2,0):0, (2,1):2, (2,2):3, (2,3):2,
            (3,0):0, (3,1):3, (3,2):2, (3,3):1,
        }
        gf4_inv = {1: 1, 2: 3, 3: 2}
        
        def gf4_add(a, b):
            return a ^ b
        
        def gf4_mul(a, b):
            return gf4_mult[(a, b)]
        
        def gf4_div(a, b):
            return gf4_mul(a, gf4_inv[b])
        
        best_colors = []
        for i in range(n):
            ai, bi = i // 4, i % 4
            for j in range(i + 1, n):
                aj, bj = j // 4, j % 4
                d1 = gf4_add(ai, aj)
                d2 = gf4_add(bi, bj)
                
                if d1 == 0 and d2 == 0:
                    c = 0
                elif d1 == 0:
                    c = 0
                elif d2 == 0:
                    c = 1
                else:
                    slope = gf4_div(d2, d1)
                    if slope == 1:
                        c = 2
                    elif slope == 2:
                        c = 3
                    else:
                        c = 4
                best_colors.append(c)
        best_n = n

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
