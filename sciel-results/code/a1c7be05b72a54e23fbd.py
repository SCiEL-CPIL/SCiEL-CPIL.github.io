"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import collections

def construct(k, g):
    deadline = time.time() + 115
    if k != 4:
        # Fallback for other k, though task specifies k=4
        return _luw_construct(k, g, deadline)
    
    # Moore bound for k=4, g=10 (even girth 2r, r=5)
    # M(4, 10) = 4 * (1 + 3 + 9 + 27 + 81) = 484
    # But prompt says lower bound 243. Let's start from 243 to be safe?
    # Actually, 484 is the Moore bound. If we start from 243, we might waste time.
    # However, if a graph exists with 320, it's below 484.
    # This implies the Moore bound calculation might be different or I am misremembering.
    # Moore bound for regular graph of degree k and girth g:
    # If g is even, g=2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
    # For k=4, r=5: 4 * (1 + 3 + 9 + 27 + 81) = 484.
    # Why does the prompt say 243?
    # Maybe the graph is not required to be connected? "Construct a 4-regular graph". Usually implies connected.
    # Or maybe the lower bound 243 is from a different paper/bound.
    # Let's start from n_min = 243 as per prompt's hint, but verify.
    
    n_min = 243
    best_n = None
    best_edges = None
    
    for n in range(n_min, n_min + 100): # Search a small range
        if time.time() > deadline - 5:
            break
            
        # Try to generate a random 4-regular graph of size n
        # We need n to be even? No, sum of degrees 4n is always even.
        # But for 4-regular, n can be any integer >= 4?
        # Actually, if n is odd, 4n is divisible by 4? No.
        # Sum of degrees is 4n. This is always even.
        # But can we have a 4-regular graph on odd n?
        # Yes, e.g., K5 is 4-regular.
        
        edges = _generate_random_4_reg(n)
        if edges is None:
            continue
            
        if _has_girth_at_least(edges, n, g):
            best_n = n
            best_edges = edges
            break # Found the smallest one in this range
            
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback: Return LUW result if it's better than nothing, or a larger random one
    # If we didn't find one in [243, 343], try a larger one quickly
    for n in range(344, 500):
        if time.time() > deadline - 5:
            break
        edges = _generate_random_4_reg(n)
        if edges is not None and _has_girth_at_least(edges, n, g):
            return n, edges
            
    # If all else fails, return the LUW construction (slow but valid)
    return _luw_construct(k, g, deadline)

def _generate_random_4_reg(n):
    """Generate a random simple 4-regular graph on n vertices."""
    if n < 4:
        return None
        
    # Configuration Model
    # Create 4n stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
            
    # Shuffle
    random.shuffle(stubs)
    
    # Pair up stubs
    edges_set = set()
    edges = []
    valid = True
    for i in range(0, 4*n, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            valid = False
            break
        if (u, v) in edges_set or (v, u) in edges_set:
            valid = False
            break
        edges_set.add((min(u, v), max(u, v)))
        edges.append((u, v))
        
    if not valid:
        return None
        
    return edges

def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g."""
    # Build adjacency list
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for u in range(n):
        # dist[v] = distance from u
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = collections.deque([u])
        
        while queue:
            x = queue.popleft()
            # If we have explored deeper than g/2, we can stop for this u?
            # No, we need to check all edges.
            # But we only care about cycles < g.
            # Max depth needed is g//2 - 1?
            # If g=10, we check up to depth 4.
            # If we find a cycle of length L < 10, it must be detected within depth 4.
            if dist[x] >= g // 2:
                continue
                
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    parent[y] = x
                    queue.append(y)
                else:
                    # Back edge or cross edge
                    if y != parent[x]:
                        # Cycle length = dist[x] + dist[y] + 1
                        # But this formula works if y is an ancestor?
                        # Actually, in BFS, if y is already visited,
                        # the cycle length is dist[x] + dist[y] + 1 ONLY IF
                        # the path from u to y does not go through x.
                        # Since BFS is level-order, if dist[y] < dist[x],
                        # y is an ancestor or in a previous level.
                        # The shortest path from x to y is 1 (edge x-y).
                        # The path from u to x is dist[x].
                        # The path from u to y is dist[y].
                        # The cycle is u->...->y->x->...->u?
                        # No, the cycle is formed by the tree path from u to x,
                        # the edge x-y, and the tree path from y to u.
                        # Length = dist[x] + 1 + dist[y].
                        # This is valid if the paths u->x and u->y are disjoint except at u.
                        # In a BFS tree, they share the path u->...->lca(x,y).
                        # So the cycle length is dist[x] + dist[y] - 2*dist[lca] + 1.
                        # This is <= dist[x] + dist[y] + 1.
                        # So if dist[x] + dist[y] + 1 < g, we definitely have a short cycle.
                        # If dist[x] + dist[y] + 1 >= g, we might still have a short cycle
                        # if lca is deep? No, lca depth reduces the length.
                        # So if dist[x] + dist[y] + 1 < g, it's a valid detection.
                        # If dist[x] + dist[y] + 1 >= g, we can't conclude it's long.
                        # However, for girth >= g, we need to ensure NO cycle is < g.
                        # So if we find dist[x] + dist[y] + 1 < g, return False.
                        
                        # Wait, is it possible that dist[x] + dist[y] + 1 >= g
                        # but the actual cycle is shorter?
                        # Yes, if lca is not u.
                        # Example: u is root. x at depth 4, y at depth 4.
                        # Edge x-y. Cycle u->...->x->y->...->u.
                        # If lca is u, length 9.
                        # If lca is at depth 2, length 4+4-4+1 = 5.
                        # So we MUST check the actual cycle length.
                        
                        # To be safe and simple:
                        # If dist[x] + dist[y] + 1 < g:
                        #    return False
                        # This is a sufficient condition for "short cycle exists".
                        # Is it necessary?
                        # If a cycle of length L < g exists, will it be detected?
                        # Yes, by BFS from any node on the cycle.
                        # Let u be a node on the cycle.
                        # The cycle has length L.
                        # BFS from u will find the "other side" of the cycle.
                        # The distance from u to the other node v on the cycle via the tree
                        # will be at most L/2.
                        # The edge connecting them will give dist[x] + dist[y] + 1.
                        # Actually, if we run BFS from u, and the cycle is u-v1-v2-...-vk-u.
                        # BFS will find v at depth d.
                        # If we encounter an edge (x, y) where x and y are on the cycle,
                        # and x is at depth d1, y at depth d2.
                        # If the edge is "back" in the cycle, d1 + d2 + 1 might be L.
                        # So if L < g, then d1 + d2 + 1 < g.
                        # So the check `if dist[x] + dist[y] + 1 < g: return False` is correct.
                        
                        if dist[x] + dist[y] + 1 < g:
                            return False
                            
    return True

def _luw_construct(k, g, deadline):
    # Original LUW code, kept as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    for idx, vertex in enumerate(order):
        if time.time() > deadline - 10:
            break
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((idx, index[other]))
                
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
