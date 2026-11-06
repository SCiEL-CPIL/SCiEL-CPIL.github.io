import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def get_edge_index(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def get_edge_vertices(idx, n):
        # Inverse of get_edge_index
        # i * (2n - i - 1) / 2 <= idx < (i+1) * (2n - i - 2) / 2
        # Approximate i
        # i^2 - i(2n-1) + 2idx < 0
        # i = (2n-1 - sqrt((2n-1)^2 - 8idx)) / 2
        # Let's just iterate or use a precomputed list for small n
        pass # We will precompute edge list

    def solve_n(n, time_limit):
        # Precompute edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        colors = [0] * num_edges
        # adj[c][u] is a bitmask of neighbors of u with color c
        adj = [[0] * n for _ in range(4)]
        
        # Symmetry breaking map: edge index -> forced color
        # (0,1) is idx 0
        # (0,2) is idx 1
        # (0,3) is idx 2 (if n>=4)
        forced_colors = {}
        if n >= 2: forced_colors[0] = 0
        if n >= 3: forced_colors[1] = 1
        if n >= 4: forced_colors[2] = 2

        def check_conflict(u, v, c):
            # Check if there is a path of length 5 from u to v in adj[c]
            # BFS
            if u == v: return False # Not a simple cycle
            # We want path u -> x1 -> x2 -> x3 -> x4 -> v
            # Depth 0: u
            # Depth 1: neighbors of u
            # ...
            # Depth 5: v
            
            # Use a queue of (node, depth)
            # Since n is small, we can use a simple visited set or bitmask
            # visited_mask
            visited = 1 << u
            frontier = [u]
            
            for depth in range(5):
                next_frontier = []
                for node in frontier:
                    # Get neighbors in color c
                    neighbors_mask = adj[c][node]
                    # Iterate through set bits
                    m = neighbors_mask
                    while m:
                        lsb = m & -m
                        nxt = lsb.bit_length() - 1
                        m ^= lsb
                        
                        if nxt == v and depth + 1 == 5:
                            return True
                            
                        if not (visited & (1 << nxt)):
                            visited |= (1 << nxt)
                            next_frontier.append(nxt)
                frontier = next_frontier
                if not frontier:
                    break
            return False

        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False # Timeout signal
            
            if idx == num_edges:
                return True
                
            u, v = edges[idx]
            
            # Determine available colors
            if idx in forced_colors:
                candidates = [forced_colors[idx]]
            else:
                candidates = [0, 1, 2, 3]
                
            # Optional: Sort candidates by some heuristic? 
            # Maybe colors with fewer conflicts first? 
            # For now, try all.
            
            for c in candidates:
                if not check_conflict(u, v, c):
                    # Assign
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                        
                    # Undo
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    
            return False

        if backtrack(0):
            return colors
        return None

    # Try n from 17 down to 10
    # Allocate time based on n
    # n=17: 60s, n=16: 20s, n=15: 10s, n=14: 5s, n=13: 2s, n=12: 1s, n=11: 0.5s, n=10: 0.2s
    time_alloc = {17: 60, 16: 20, 15: 10, 14: 5, 13: 2, 12: 1, 11: 0.5, 10: 0.2}
    
    for n in range(17, 9, -1):
        if time.time() > hard_deadline - 2:
            break
        t_limit = time_alloc.get(n, 1)
        # Ensure we don't exceed hard deadline
        remaining = hard_deadline - time.time() - 2
        t_limit = min(t_limit, remaining)
        
        if t_limit <= 0:
            continue
            
        res = solve_n(n, t_limit)
        if res:
            # Verify (optional, but good for safety)
            # For now, trust the solver
            return {'n': n, 'colors': res}
            
    # Fallback: n=9
    # We can use a known valid coloring or just return a random one that we verify
    # Let's try to find a valid one for n=9 quickly with random search
    n = 9
    num_edges = n * (n - 1) // 2
    # Random search with verification
    def verify(n, colors):
        # Build adj
        adj = [[0]*n for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                idx = get_edge_index(i, j, n)
                c = colors[idx]
                adj[c][i] |= 1 << j
                adj[c][j] |= 1 << i
                
        # Check for C6 in each color
        for c in range(4):
            # Check all pairs (u, v)
            for u in range(n):
                for v in range(u+1, n):
                    if adj[c][u] & (1 << v):
                        # Check if path of length 5 exists
                        visited = 1 << u
                        frontier = [u]
                        for depth in range(5):
                            next_frontier = []
                            for node in frontier:
                                m = adj[c][node]
                                while m:
                                    lsb = m & -m
                                    nxt = lsb.bit_length() - 1
                                    m ^= lsb
                                    if nxt == v and depth + 1 == 5:
                                        return False # Conflict found
                                    if not (visited & (1 << nxt)):
                                        visited |= 1 << nxt
                                        next_frontier.append(nxt)
                            frontier = next_frontier
                            if not frontier: break
        return True

    # Try a few random colorings for n=9
    for _ in range(1000):
        if time.time() > hard_deadline - 1:
            break
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        if verify(n, colors):
            return {'n': n, 'colors': colors}
            
    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
