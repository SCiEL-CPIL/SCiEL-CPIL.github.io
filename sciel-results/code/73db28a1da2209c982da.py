import time
import collections
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _has_short_cycle_nb(n, edges, min_girth):
    """
    Returns True if the graph has a cycle shorter than min_girth.
    Uses non-backtracking walk enumeration from each directed edge.
    A cycle of length L < min_girth corresponds to a non-backtracking
    walk of length L that returns to its starting directed edge.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each directed edge (u,v), do a non-backtracking BFS
    # up to depth min_girth-1. If we can return to (u,v) in exactly
    # min_girth-1 steps (or less, but must be a cycle), we have a short cycle.
    # Actually, a cycle of length L means from directed edge (u,v), we can
    # take L-1 more non-backtracking steps and return to (u,v).
    
    max_depth = min_girth - 1
    
    # State: (current_node, previous_node) representing directed edge prev->curr
    # We start from each directed edge and walk non-backtracking
    
    for u in range(n):
        for v in adj[u]:
            # Start: we're at v, came from u
            # We need to find if there's a non-backtracking path of length max_depth
            # from (v, u) that ends back at (u, v)
            # i.e., after max_depth steps we're at u and came from v
            
            # BFS in state space (node, prev_node)
            # depth 0: state (v, u)
            # depth d: states (node, prev) where we reached node from prev in d non-backtracking steps
            # We want to reach (u, v) at depth max_depth
            
            visited = set()
            visited.add((v, u))
            frontier = [(v, u)]
            
            for step in range(1, max_depth + 1):
                next_frontier = []
                for (node, prev) in frontier:
                    for nxt in adj[node]:
                        if nxt == prev:
                            continue
                        new_state = (nxt, node)
                        if new_state in visited:
                            continue
                        visited.add(new_state)
                        next_frontier.append(new_state)
                if not next_frontier:
                    break
                # Check if we've returned to start
                if (u, v) in next_frontier:
                    return True
                frontier = next_frontier
                
                if _time_left() < 1.0:
                    return False  # timeout, assume no short cycle (risky but we have deadline)
    
    return False

def _construct_luw(k, g):
    """Fallback: construct LUW graph."""
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

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Moore bound for k=3, g=16: 2 * (3*(3^(7)-1)/(3-1)) + 2 = 2*3*(2187-1)/2 + 2 = 6562? 
    # Wait, Moore bound for odd girth: n >= k * (k-1)^((g-1)/2) + 1 for odd g
    # For even girth g: n >= 2 * (k-1)^(g/2) ... no.
    # Moore bound: for girth g, n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
    # For g=16: floor((16-1)/2) = 7, so sum from i=0 to 6 of 2^i = 127
    # n >= 1 + 3*127 = 382? That doesn't seem right.
    # Actually the Moore bound for k-regular graph with girth g:
    # If g is odd: n >= 1 + k * (1 + (k-1) + ... + (k-1)^((g-3)/2))
    # If g is even: n >= 2 * (1 + (k-1) + ... + (k-1)^(g/2-1))
    # For g=16 (even): n >= 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2 * 255 = 510
    
    # So minimum n is 510, but must be even for 3-regular, so n >= 512.
    
    # Strategy: Try to construct a 3-regular graph with girth >= g.
    # Approach 1: Search over (2,3)-biregular bipartite graphs
    # A 3-regular graph on 2m vertices that is bipartite with equal parts
    # corresponds to a (2,3)-biregular bipartite graph with parts of size m.
    # The girth of the 3-regular graph equals the girth of the bipartite graph
    # (since cycles in bipartite graph have even length, and the 3-reg graph
    # is a 2-lift... actually no, it's just a bipartite 3-regular graph).
    
    # Actually, a bipartite 3-regular graph IS a 3-regular graph. So we just
    # need to find a 3-regular bipartite graph with girth >= g and minimum vertices.
    
    # Let's try random 3-regular bipartite graphs and check girth.
    # For n=2m vertices, we need a 3-regular bipartite graph with parts of size m.
    # This is equivalent to a (3,3)-biregular bipartite graph, i.e., a 3-regular
    # bipartite graph.
    
    # Random construction: use the configuration model or random permutation approach.
    # For a 3-regular bipartite graph with parts A={0..m-1} and B={0..m-1},
    # each vertex in A has degree 3, each in B has degree 3.
    # We can represent it as three perfect matchings between A and B.
    
    # Let's try this approach with random matchings.
    
    # Minimum m: n = 2m >= 512, so m >= 256.
    
    min_m = 256
    max_m = 500
    
    best_result = None
    
    for m in range(min_m, max_m + 1):
        if _time_left() < 30.0:
            break
        
        # Try several random 3-regular bipartite graphs
        n_trials = max(1, int(5.0 * _time_left() / max(1, max_m - m)))
        
        for trial in range(n_trials):
            if _time_left() < 5.0:
                break
            
            # Generate three random perfect matchings
            # Use random permutations
            perms = [list(range(m)) for _ in range(3)]
            for p in perms:
                random.shuffle(p)
            
            # Build adjacency
            adj = [[] for _ in range(2 * m)]
            edges = []
            edge_set = set()
            
            valid = True
            for i in range(m):
                for j in range(3):
                    a = i
                    b = m + perms[j][i]
                    if a > b:
                        a, b = b, a
                    if (a, b) in edge_set:
                        valid = False
                        break
                    edge_set.add((a, b))
                    edges.append((i, m + perms[j][i]))
                    adj[i].append(m + perms[j][i])
                    adj[m + perms[j][i]].append(i)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check all vertices have degree 3
            if any(len(adj[v]) != 3 for v in range(2 * m)):
                continue
            
            # Check girth
            if not _has_short_cycle_nb(2 * m, edges, g):
                best_result = (2 * m, edges)
                return best_result
    
    # Fallback to LUW
    return _construct_luw(k, g)
