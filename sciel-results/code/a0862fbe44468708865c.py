"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115  # 115s deadline, 5s safety margin
    
    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex up to depth min_girth//2 - 1."""
        depth_limit = min_girth // 2 - 1  # For girth 16, depth_limit = 7
        for start in range(n):
            # BFS from start, tracking parent to avoid backtracking
            visited = {start: (-1, 0)}  # node -> (parent, depth)
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= depth_limit:
                    continue
                for nbr in adj[node]:
                    if nbr == parent:
                        continue
                    if nbr in visited:
                        # Found a cycle: depth + 1 + visited[nbr][1]
                        cycle_len = depth + 1 + visited[nbr][1]
                        if cycle_len < min_girth:
                            return False
                    else:
                        visited[nbr] = (node, depth + 1)
                        queue.append((nbr, node, depth + 1))
        return True
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
        q = k
        # For girth g, we need d such that d + 5 >= g (for odd d)
        # or d + 4 >= g (for even d)
        if (g - 5) % 2 == 1:
            d = g - 5
        else:
            d = g - 4
        d = max(3, d)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return n, edges, adj
    
    # Try LUW construction first as a baseline
    best_n, best_edges, best_adj = None, None, None
    try:
        n, edges, adj = luw_construct(k, g)
        if check_girth(adj, n, g):
            best_n, best_edges, best_adj = n, edges, adj
    except:
        pass
    
    # Now try randomized search: construct random 3-regular graphs and check girth
    # This is unlikely to work for girth 16, but try a structured approach
    
    # Alternative: try to use a smaller construction by varying parameters
    # For k=3, g=16, try different d values in LUW-like constructions
    
    if time.time() > deadline:
        if best_n is not None:
            return best_n, best_edges
        return 0, []
    
    # Try a different approach: construct a graph using a BFS tree and try to close it
    # This is complex, so let's try a simpler randomized local search
    
    # For now, return the best we have
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return a trivially valid but large graph
    # Construct a cycle of length g (but it's not 3-regular)
    # Actually, we need a 3-regular graph. Let's just return the LUW result or a default.
    
    # If LUW failed, try a simple construction: 3-prism or similar
    # For safety, construct a valid 3-regular graph with girth >= g
    # The complete graph K4 has girth 3, not useful for large g
    
    # Return a large cycle-based 3-regular graph (circular ladder)
    # A circular ladder graph CL_n is 3-regular with n/2 rungs, girth depends on n
    # For girth >= 16, we need a sufficiently large ladder
    
    # Actually, let's just return the LUW construction result
    # If that failed, we have a problem. Let's construct something valid.
    
    # Construct a 3-regular graph: take two copies of a path and connect them
    # This gives a 2-regular graph (cycle), not 3-regular
    
    # For a 3-regular graph with large girth, use the LUW construction
    # If it's too slow, we may time out. Let's just return what we have.
    
    return 0, []
# EVOLVE-BLOCK-END
