# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for efficient,
    verifiable high-girth regular graphs.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin against 120s limit

    if k != 3:
        # Fallback for other degrees, though task specifies k=3
        return _lnw_construct(k, g)

    # We want a 3-regular graph.
    # LUW graphs are q-regular. So we set q=k=3.
    # The number of vertices is 2 * q^d.
    # We search for the smallest d such that girth(L_{3,d}) >= g.
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound check: n >= 2 * (1 + (k-1)^(g/2 - 1))
    # For k=3, g=16: n >= 2 * (1 + 2^7) = 512.
    
    # Try increasing d. d determines the "height" or number of columns.
    # d=5 -> n=486 (too small, girth max 512 bound implies girth < 16 likely)
    # d=6 -> n=1458
    # d=7 -> n=4374
    
    # Start from d=5 to be safe, but effectively d=6 or 7 will be the first valid.
    for d in range(5, 15):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _lnw_construct_params(k, d)
        except Exception as e:
            # If construction fails, skip to next d
            continue
            
        if n >= best_n:
            continue
            
        # Verify the graph properties
        # 1. Simple and k-regular
        if not _is_valid_regular_graph(edges, n, k):
            continue
            
        # 2. Girth
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            if n < best_n:
                best_n = n
                best_edges = edges
            # Since n increases with d, the first valid d is the best.
            # We can break here.
            break

    if best_edges is not None:
        return best_n, best_edges

    # Absolute fallback: A very large d is guaranteed to have high girth.
    # Though it will have many vertices, it ensures a non-zero score if the above fails.
    d = 12
    try:
        return _lnw_construct_params(k, d)
    except Exception:
        # Last resort: empty graph or trivial? No, must be valid.
        # If LUW fails completely, we can construct a cycle of length g? No, degree 3.
        # A large complete graph? No, not regular with fixed k.
        # Just return a known large d LUW.
        return _lnw_construct_params(k, 15)

def _is_valid_regular_graph(edges, n, k):
    """
    Check if the graph is simple, k-regular.
    """
    if n == 0:
        return k == 0
    
    deg = [0] * n
    seen = set()
    
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        # Ensure unique edges
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
        
    for i in range(n):
        if deg[i] != k:
            return False
            
    return True

def _edges_to_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _calculate_girth(adj, n):
    """
    Calculate the girth (length of the shortest cycle) using BFS from each vertex.
    Optimized by early termination if we find a cycle shorter than the current best.
    """
    min_cycle = n + 1
    
    for start in range(n):
        # Early exit if we already have a very short cycle
        if min_cycle <= 2: 
            return min_cycle
            
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is not the parent, we found a cycle
                    if v != parent[u]:
                        # The cycle length is dist[u] + dist[v] + 1
                        # But we must ensure it's a simple cycle?
                        # In BFS on undirected graphs, any edge (u,v) where dist[u] != dist[v]+1 or dist[v] != dist[u]+1 
                        # or generally a back edge creates a cycle.
                        # Specifically, if dist[u] + dist[v] + 1 < min_cycle, we update.
                        # Note: BFS tree edges connect dist d to d+1.
                        # Non-tree edges connect same or adjacent levels (or far, but shortest path logic applies).
                        # The length of the cycle formed by u-v and their paths to LCA is dist[u]+dist[v]+1 - 2*dist[LCA].
                        # However, a simpler heuristic for girth in sparse regular graphs:
                        # Just take dist[u] + dist[v] + 1? No, that's the length of the closed walk.
                        # If u and v are not parent-child, the path u->start + start->v + edge is a closed walk.
                        # The actual cycle might be shorter if they share a prefix.
                        # However, for the purpose of finding the *shortest* cycle, 
                        # if dist[u]+dist[v]+1 is small, the cycle is likely small.
                        # To be exact, we should check LCA, but that's slow.
                        # Standard approximation: In BFS, the first back edge found from a node at depth d 
                        # to a node at depth d or d-1 (not parent) forms a cycle of length <= 2d+1.
                        # Let's just use the closed walk length as an upper bound check?
                        # Actually, for girth, we need the *exact* shortest cycle.
                        # If we just return dist[u]+dist[v]+1, it might be an overestimate if there's a common ancestor.
                        # But if we find a cycle of length L, and L < min_cycle, we update.
                        # If the true cycle is shorter, we might miss it?
                        # No, BFS explores in order of distance. The shortest cycle will be detected 
                        # when the BFS frontier meets.
                        # Let's stick to the standard: cycle_len = dist[u] + dist[v] + 1.
                        # This is valid if u and v are not connected by a tree edge.
                        # This gives the length of the cycle formed by the two unique paths from start to u and v.
                        # This is a valid cycle in the graph. It might not be the *shortest* cycle in the graph, 
                        # but if it's < min_cycle, we update min_cycle.
                        # Eventually, the true shortest cycle will be found by some start node.
                        
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    """
    Generate the Lazebnik-Ustimenko-Woldar graph L_{k,d}.
    k is the degree (q).
    d is the parameter (number of columns in the standard representation).
    """
    q = k
    # For q=2, use XOR and specific table? No, standard LUW works for any q >= 2.
    # Arithmetic mod q.
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    # Rule table for LUW
    # rule[c] = (a, b) such that y[c] = x[c] + y[a] * x[b] (for side 0)
    # Standard rule:
    # c=2: y[2] = x[2] + y[1] * x[1]  => (1, 1)
    # c=3: y[3] = x[3] + y[2] * x[1]  => (2, 1)
    # c=4: y[4] = x[4] + y[3] * x[1]  => (3, 1) ... wait, standard is often (c-1, 1) for c>2?
    # Let's check the provided seed code's rule generation.
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # I will replicate the exact rule from the seed to ensure compatibility with known good instances.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple (x[0] is dummy/None, x[1]..x[d] are values)
        # Length of x is d+1.
        # y[0] is determined by 'first' (which is x'[1] essentially? No.)
        # In LUW, the neighbors of (0, x) are (1, y) such that:
        # y[1] can be any value? No.
        # The construction:
        # V = { (0, x) | x in Q^d } U { (1, y) | y in Q^d }
        # Actually, usually x[0] is not used, or x is length d.
        # Seed code: `start = (0, (None,) + (0,) * d)` -> length d+1.
        # `y = [None, first] + [0] * (d - 1)` -> length d+1.
        # So x[1]..x[d] are the coordinates. x[0] is None.
        
        for first in range(q):
            # 'first' corresponds to the neighbor's first coordinate?
            # In the seed: `y = [None, first] ...`
            # So for side 0 to 1, the new vertex has y[1]=first.
            # And for each such 'first', we calculate the rest.
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # side 0: y[c] = x[c] + y[a] * x[b]
                # side 1: y[c] = x[c] - x[a] * y[b]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _lnw_construct(k, g):
    # Fallback helper if called directly
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
