import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Determine minimum d for LUW
    # Girth of LUW CD(d, k) is at least d + 5
    # So we need d + 5 >= g => d >= g - 5
    min_d = max(3, g - 5)
    
    # Try LUW
    n, edges = _luw_construct_fast(k, min_d)
    
    # Verify
    if _verify_girth_fast(n, edges, g):
        return n, edges
        
    # Fallback: larger d
    n, edges = _luw_construct_fast(k, min_d + 1)
    return n, edges

def _luw_construct_fast(k, d):
    q = k
    # Precompute GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute rule
    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
            
    n_vertices = 2 * (q ** d)
    q_pow_d = q ** d
    
    # Helper to convert integer to vector
    def int_to_vec(idx):
        vec = [0] * d
        val = idx
        for i in range(d - 1, -1, -1):
            vec[i] = val % q
            val //= q
        return vec
        
    # Helper to convert vector to integer
    def vec_to_int(vec):
        val = 0
        for i in range(d):
            val = val * q + vec[i]
        return val

    edges = []
    
    # Iterate over all vertices in side 0
    for i in range(q_pow_d):
        x_vec = int_to_vec(i)
        # x is 1-indexed in the formula: x[1]..x[d]
        # In our vec, indices are 0..d-1 corresponding to 1..d
        
        for first in range(q):
            # y[1] = first
            y_vec = [0] * d
            y_vec[0] = first # y[1]
            
            # Calculate y[2]..y[d]
            for c in range(2, d + 1):
                a, b = rule[c]
                # x[a], x[b], y[a] are 1-indexed
                # In 0-indexed: x[a-1], x[b-1], y[a-1]
                if i < q_pow_d: # side 0
                    # y[c] = x[c] + y[a] * x[b]
                    # c is 1-indexed, so c-1 in vec
                    val = add(x_vec[c-1], mul(y_vec[a-1], x_vec[b-1]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    val = sub(x_vec[c-1], mul(x_vec[a-1], y_vec[b-1]))
                y_vec[c-1] = val
                
            y_idx = vec_to_int(y_vec)
            other_idx = q_pow_d + y_idx # side 1
            
            edges.append((i, other_idx))
            
    return n_vertices, edges

def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from every vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

# Keep the original functions for compatibility if needed, but the fast ones are used
def _luw_construct(k, d):
    return _luw_construct_fast(k, d)

def _verify_girth(n, edges, g):
    return _verify_girth_fast(n, edges, g)

def _construct_5_9(deadline):
    pass

def _random_5_regular(n, rng):
    pass

def _get_adjacency(edges, n):
    pass

def _find_short_cycle(adj, n, max_len):
    pass

def _improve_girth(edges, n, target_girth, deadline, rng):
    pass

def _estimate_girth(adj, n, upper_bound):
    pass

def _has_cycle_of_length(adj, n, length):
    pass
