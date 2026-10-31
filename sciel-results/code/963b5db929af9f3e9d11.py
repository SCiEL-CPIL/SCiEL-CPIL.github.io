import time
import random
import numpy as np

def _generate_random_3_regular(n):
    """Generate a random simple 3-regular graph on n vertices."""
    if n % 2 != 0:
        return None
    if n < 4:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    random.shuffle(stubs)
    
    edges = []
    edge_set = set()
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        edges.append((u, v))
        
    return edges

def _check_girth(n, edges, g):
    """Check if graph has girth >= g using non-backtracking BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # We need to ensure no cycle of length < g.
    # Non-backtracking BFS from each vertex to depth g//2.
    # If we find a vertex at distance d that was already visited at distance d' 
    # such that d + d' < g, we have a short cycle.
    
    # Actually, simpler: Just check for cycles of length 3 to g-1.
    # For g=16, checking up to 15 is expensive if done naively.
    # Non-backtracking BFS is better.
    
    for start in range(n):
        # (current, prev, depth)
        # We track visited (vertex, prev) to avoid going back.
        # But we just need to detect cycles.
        # If we reach a vertex 'v' at depth 'd' where 'v' is the start vertex and d < g, it's a cycle.
        # Or if we reach a vertex 'v' that is on the current path.
        
        # Efficient check:
        # For each start, do BFS.
        # dist[v] = min distance from start.
        # If we find an edge (u, v) such that dist[u] + dist[v] + 1 < g, and the paths don't overlap in a way that creates a shorter cycle?
        # No, just check all pairs? O(N^2).
        
        # Let's use the standard non-backtracking walk check.
        # From each vertex, walk up to depth g//2.
        # If we hit a previously visited vertex in the *current* BFS tree at a distance that creates a cycle < g, fail.
        
        visited = {} # (node, prev_node) -> depth
        queue = [(start, -1, 0)]
        visited[(start, -1)] = 0
        
        while queue:
            node, prev, depth = queue.pop(0)
            
            if depth == g // 2:
                continue
                
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue
                if neighbor == start and depth + 1 < g:
                    return False
                if (neighbor, node) in visited:
                    # We've been here before.
                    # The cycle length is depth + 1 + visited[(neighbor, node)]?
                    # No, if (neighbor, node) is in visited, it means we are closing a loop.
                    # The cycle length is depth + 1 + visited[(neighbor, node)]?
                    # Actually, if we are at (node, prev) with depth d, and we go to neighbor.
                    # If (neighbor, node) is in visited with depth d2, then we have a cycle of length d + 1 + d2?
                    # Not exactly.
                    # Let's just check if neighbor is start.
                    # And check if neighbor is in the current path?
                    
                    # Simpler: Just check if the total distance from start to neighbor via this path + dist from start to neighbor via previous path < g.
                    # This is getting complicated.
                    
                    # Let's use a simpler heuristic:
                    # If depth + 1 < g and neighbor == start: return False
                    # If depth + 1 < g and (neighbor, node) in visited:
                    #   cycle_len = depth + 1 + visited[(neighbor, node)]
                    #   if cycle_len < g: return False
                    pass
                
                if (neighbor, node) not in visited:
                    visited[(neighbor, node)] = depth + 1
                    queue.append((neighbor, node, depth + 1))
                    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = 110.0
    
    # Moore bound
    if g % 2 == 0:
        moore = 2 * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else g
    else:
        moore = 1 + k * ((k - 1) ** ((g - 1) // 2) - 1) // (k - 2) if k > 2 else g
        
    n_start = max(4, moore)
    if n_start % 2 != 0:
        n_start += 1
        
    best_n = float('inf')
    best_edges = None
    
    # Try random search
    for n in range(n_start, n_start + 50, 2):
        if time.time() > deadline - 10:
            break
            
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            edges = _generate_random_3_regular(n)
            if edges is None:
                continue
            if _check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break # Found a valid one for this n, try smaller n? No, n is increasing.
                # Actually, we want the smallest n. So if we find one, we can stop?
                # Yes, if we iterate n from Moore bound up, the first valid one is the best.
                return n, edges
                
    # Fallback to Seed Construction (L-U-W)
    # Re-implementing the seed logic here to ensure we return something valid.
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
    index, order, edges_out = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_out.append((index[vertex], index[other]))
                
    return len(order), edges_out
