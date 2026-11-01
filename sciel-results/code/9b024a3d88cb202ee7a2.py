import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    if k != 5:
        return _luw_construct(k, g, deadline)

    # For k=5, g=9: try LUW with different d values, pick smallest valid
    # d must satisfy: girth >= d + 5 >= g, so d >= g - 5
    # But the LUW construction has specific parity requirements
    # For g=9: d >= 4, but d must be odd for the standard construction
    # Actually, let's just try d values and check girth
    
    best_n = None
    best_edges = None
    
    # Try d values from minimum valid upward
    # For girth g, we need d + 5 >= g, so d >= g - 5
    # The construction works for d >= 2
    # Let's try d = g-5, g-4, g-3, ... and pick the one that gives smallest n
    # But we also need to verify girth >= g
    
    min_d = g - 5
    if min_d < 2:
        min_d = 2
    
    # Try a range of d values
    for d in range(min_d, min_d + 10):
        if time.time() > deadline - 20.0:
            break
        
        try:
            n, edges = _luw_construct(5, d, deadline)
            if n is None:
                continue
            
            # Verify girth >= g using non-backtracking BFS
            if _verify_girth(edges, n, g, deadline):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            continue
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the standard LUW with the original d calculation
    return _luw_construct(5, g, deadline)


def _luw_construct(k, g, deadline):
    """LUW construction for k-regular graph with girth at least g.
    
    Uses d parameter where girth >= d + 5.
    For target girth g, use d = g - 5 (or adjusted for parity).
    """
    q = k
    # d parameter: for girth >= g, need d + 5 >= g
    # Standard choice: d = g - 5 if (g-5) is odd, else g - 4
    # This ensures the construction works properly
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute multiplication table
    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q
    
    # Build the rule table for LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)
    
    free_vals = np.arange(q, dtype=np.int64)
    
    def get_neighbours(side, x_arr):
        """Compute all neighbours of a vertex using vectorized operations."""
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[y_coords[:, a], x_arr[b]]
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[x_arr[a], y_coords[:, b]]
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords
    
    # BFS to build the graph
    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((index[vertex], index[other_key]))
    
    return len(order), edges


def _verify_girth(edges, n, g, deadline):
    """Verify that the graph has girth at least g using BFS from each vertex.
    
    Checks for cycles of length 3 to g-1.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth >= g, we need no cycles shorter than g
    # BFS from each vertex up to depth floor((g-1)/2)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        if time.time() > deadline - 3.0:
            return False  # Can't verify in time, assume invalid
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= max_depth:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    # Found a potential cycle
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    
    return True
