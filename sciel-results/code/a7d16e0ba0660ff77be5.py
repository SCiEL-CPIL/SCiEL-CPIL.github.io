import time
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback to LW graph for other parameters
        return _construct_lw(k, g)
    
    # Voltage graph (covering) construction for 4-regular graph with girth 10
    # Base: 4-cycle C4 with vertices 0,1,2,3
    # Edges: (0,1), (1,2), (2,3), (3,0)
    # Each edge gets a voltage in Z_m
    # Lift has 4*m vertices, 4-regular
    
    # We need to find m and voltages such that:
    # 1. No cycle of length < 10 in the lift
    # 2. The graph is simple (no loops, no multi-edges)
    
    # For C4 base, closed walks of length L have voltage sums that must be nonzero mod m
    # for L < g to avoid short cycles
    
    # Try different m values and voltage assignments
    best_n = float('inf')
    best_edges = None
    
    # For k=4, g=10, we expect n around 320 (known best)
    # Try m values that give n = 4*m in a reasonable range
    for m in range(40, 81):  # n = 160 to 320
        if time.time() - start_time > deadline - 5:
            break
        
        # Base C4: edges (0,1), (1,2), (2,3), (3,0)
        # Voltage assignment: v01, v12, v23, v30 in Z_m
        # For the lift to be simple and have good girth, we need careful choices
        
        # Try a few voltage assignments
        found = False
        for attempt in range(20):
            if time.time() - start_time > deadline - 3:
                break
            
            # Generate random voltages
            v01 = random.randint(1, m-1)
            v12 = random.randint(1, m-1)
            v23 = random.randint(1, m-1)
            v30 = random.randint(1, m-1)
            
            # Ensure no trivial short cycles
            # A cycle of length 4 in base has voltage sum v01+v12+v23+v30
            # This must be nonzero mod m to avoid 4-cycles
            if (v01 + v12 + v23 + v30) % m == 0:
                continue
            
            # Check for 2-cycles (back-and-forth): voltage + (-voltage) = 0 always,
            # but these are not actual cycles in the simple graph sense
            
            # Build the lift graph
            edges = []
            n = 4 * m
            adj = [[] for _ in range(n)]
            
            # Vertex (base_vertex, group_element) -> index = base_vertex * m + group_element
            # Edge (0,1) with voltage v01: (0,a) -- (1, a+v01)
            # Edge (1,2) with voltage v12: (1,b) -- (2, b+v12)
            # Edge (2,3) with voltage v23: (2,c) -- (3, c+v23)
            # Edge (3,0) with voltage v30: (3,d) -- (0, d+v30)
            
            for a in range(m):
                # Edge (0,a) -- (1, (a+v01)%m)
                u1 = 0 * m + a
                v1 = 1 * m + (a + v01) % m
                edges.append((u1, v1))
                adj[u1].append(v1)
                adj[v1].append(u1)
                
                # Edge (1,a) -- (2, (a+v12)%m)
                u2 = 1 * m + a
                v2 = 2 * m + (a + v12) % m
                edges.append((u2, v2))
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                # Edge (2,a) -- (3, (a+v23)%m)
                u3 = 2 * m + a
                v3 = 3 * m + (a + v23) % m
                edges.append((u3, v3))
                adj[u3].append(v3)
                adj[v3].append(u3)
                
                # Edge (3,a) -- (0, (a+v30)%m)
                u4 = 3 * m + a
                v4 = 0 * m + (a + v30) % m
                edges.append((u4, v4))
                adj[u4].append(v4)
                adj[v4].append(u4)
            
            # Check if graph is simple (no duplicate edges)
            edge_set = set()
            is_simple = True
            for u, v in edges:
                if u == v:
                    is_simple = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    is_simple = False
                    break
                edge_set.add(key)
            
            if not is_simple:
                continue
            
            # Check degree
            valid = True
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check girth using non-backtracking BFS
            if _verify_girth_fast(n, adj, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    found = True
                break  # Found valid for this m, try smaller m next
        
        if found and m < 80:
            # If we found a valid graph, try to find smaller m
            pass
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LW graph if voltage construction fails
    return _construct_lw(k, g)


def _verify_girth_fast(n, adj, g):
    """Fast girth verification using non-backtracking walks."""
    for start_v in range(n):
        visited = set()
        queue = deque()
        
        for nb in adj[start_v]:
            if (nb, start_v) not in visited:
                visited.add((nb, start_v))
                queue.append((nb, start_v, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            
            if curr == start_v and depth < g:
                return False
            
            if depth >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, depth + 1))
    
    return True


def _construct_lw(k, g):
    """Fallback LW graph construction."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
