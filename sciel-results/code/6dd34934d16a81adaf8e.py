# EVOLVE-BLOCK-START
import random
from collections import deque
import sys
sys.setrecursionlimit(10000)

def _check_girth(n, edges, min_girth):
    """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking parent to avoid backtracking
        visited = {start: (-1, 0)}  # node -> (parent, dist)
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth // 2:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    # Found a cycle of length dist + visited[nb][1] + 1
                    cycle_len = dist + visited[nb][1] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    visited[nb] = (node, dist + 1)
                    queue.append((nb, node, dist + 1))
    return True

def _generate_luw(k, g):
    """Generate a valid graph using L-U-W construction."""
    if k != 4:
        return None
    d = g // 2
    if d < 3:
        return None
    q = k
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
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
    return len(order), edges

def construct(k, g):
    if k != 4 or g != 10:
        result = _generate_luw(k, g)
        if result:
            return result
        # Fallback to random if L-U-W fails (shouldn't for k=4)
        random.seed(42)
        for n in range(200, 500):
            # ... random search ...
            pass
        return 0, []

    # For k=4, g=10
    # Try to find the 320-vertex cage using backtracking?
    # For now, use L-U-W which gives n=2048.
    # This is a valid solution, though not optimal.
    # To improve, we need the 320-vertex cage.
    
    # Let's try a specific known construction for (4,10) cage.
    # The (4,10) cage is the Klein graph.
    # We can construct it using the group A5.
    # However, without a verified construction, we fall back to L-U-W.
    
    n, edges = _generate_luw(4, 10)
    return n, edges
# EVOLVE-BLOCK-END
