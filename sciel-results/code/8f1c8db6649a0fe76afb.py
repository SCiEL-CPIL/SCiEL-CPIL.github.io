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
    
    best_n = float('inf')
    best_edges = None
    
    # Try m values that give n = 4*m in a reasonable range
    # Start from smaller m to find minimum n first
    # Moore bound for (4,10) is around 243, so m >= 61 (n >= 244)
    # Known best is 320, so m = 80
    for m in range(61, 81):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try deterministic structured voltages first
        found = False
        
        # Try various voltage patterns
        patterns = []
        
        # Pattern 1: All same voltage
        for v in range(1, m):
            patterns.append((v, v, v, v))
        
        # Pattern 2: Alternating
        for v1 in range(1, m):
            for v2 in range(1, m):
                if v1 != v2:
                    patterns.append((v1, v2, v1, v2))
        
        # Pattern 3: Two pairs
        for v1 in range(1, m):
            for v2 in range(1, m):
                if v1 != v2:
                    patterns.append((v1, v1, v2, v2))
        
        # Pattern 4: Three same, one different
        for v1 in range(1, m):
            for v2 in range(1, m):
                if v1 != v2:
                    patterns.append((v1, v1, v1, v2))
                    patterns.append((v1, v1, v2, v1))
                    patterns.append((v1, v2, v1, v1))
                    patterns.append((v2, v1, v1, v1))
        
        # Pattern 5: Arithmetic progression
        for start_v in range(1, m):
            for step in range(1, m):
                patterns.append((start_v, (start_v + step) % m, (start_v + 2*step) % m, (start_v + 3*step) % m))
        
        # Pattern 6: Geometric-like
        for v1 in range(1, m):
            for v2 in range(1, m):
                patterns.append((v1, v2, (v1*v2) % m, (v1 + v2) % m))
        
        # Random patterns
        random.seed(42 + m)
        for _ in range(50):
            patterns.append((random.randint(1, m-1), random.randint(1, m-1), 
                           random.randint(1, m-1), random.randint(1, m-1)))
        
        # Deduplicate
        seen_patterns = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen_patterns:
                seen_patterns.add(p)
                unique_patterns.append(p)
        
        for v01, v12, v23, v30 in unique_patterns:
            if time.time() - start_time > deadline - 3:
                break
            
            # Ensure no 4-cycle in base
            if (v01 + v12 + v23 + v30) % m == 0:
                continue
            
            # Build the lift graph
            n = 4 * m
            adj = [[] for _ in range(n)]
            
            # Vertex (base_vertex, group_element) -> index = base_vertex * m + group_element
            for a in range(m):
                # Edge (0,a) -- (1, (a+v01)%m)
                u1 = 0 * m + a
                v1 = 1 * m + (a + v01) % m
                adj[u1].append(v1)
                adj[v1].append(u1)
                
                # Edge (1,a) -- (2, (a+v12)%m)
                u2 = 1 * m + a
                v2 = 2 * m + (a + v12) % m
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                # Edge (2,a) -- (3, (a+v23)%m)
                u3 = 2 * m + a
                v3 = 3 * m + (a + v23) % m
                adj[u3].append(v3)
                adj[v3].append(u3)
                
                # Edge (3,a) -- (0, (a+v30)%m)
                u4 = 3 * m + a
                v4 = 0 * m + (a + v30) % m
                adj[u4].append(v4)
                adj[v4].append(u4)
            
            # Check if graph is simple (no duplicate edges)
            is_simple = True
            edge_set = set()
            for i in range(n):
                for j in adj[i]:
                    if i == j:
                        is_simple = False
                        break
                    key = (min(i, j), max(i, j))
                    if key in edge_set:
                        is_simple = False
                        break
                    edge_set.add(key)
                if not is_simple:
                    break
            
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
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    found = True
                    break  # Found valid for this m, try smaller m next
        
        if found:
            break  # Found valid graph, stop searching
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LW graph if voltage construction fails
    return _construct_lw(k, g)


def _verify_girth_fast(n, adj, g):
    """Fast girth verification using non-backtracking walks."""
    for start_v in range(n):
        # Use a set of (current, previous) pairs visited
        visited = set()
        queue = deque()
        
        for nb in adj[start_v]:
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
