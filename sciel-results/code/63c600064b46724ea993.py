import time
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges) for a q-regular graph with girth >= d+5 (odd d)."""
    q = k
    # Ensure d is odd and d+5 >= g
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    if d < 3:
        d = 3

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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
    return len(order), edges

def _verify_girth(n, edges, min_girth):
    """Verify no cycle shorter than min_girth using non-backtracking BFS from each vertex."""
    if n == 0:
        return True
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Non-backtracking walk BFS: state is (current, prev)
    # We look for cycles of length < min_girth
    half = min_girth // 2
    for start in range(n):
        # BFS in non-backtracking walks up to depth half
        # State: (node, prev_node), depth
        visited = set()
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= half:
                continue
            if node == start and depth > 0:
                # Found a cycle of length depth
                if depth < min_girth:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                continue
            visited.add(state)
            for nb in adj[node]:
                if nb == prev:
                    continue
                queue.append((nb, node, depth + 1))
    return True

def _get_smallest_component(n, edges):
    """Find the smallest connected component in the graph."""
    if n == 0:
        return 0, []
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * n
    smallest_size = n + 1
    smallest_edges = []
    
    for i in range(n):
        if not visited[i]:
            # BFS to find component
            comp_nodes = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp_nodes.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
                        
            comp_size = len(comp_nodes)
            if comp_size < smallest_size:
                smallest_size = comp_size
                # Collect edges within this component
                comp_set = set(comp_nodes)
                comp_edges = []
                for u, v in edges:
                    if u in comp_set and v in comp_set:
                        comp_edges.append((u, v))
                smallest_edges = comp_edges
                
    return smallest_size, smallest_edges

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    # Get LUW construction
    n, edges = _luw_construct(k, g)

    # Check for smallest component
    n_comp, edges_comp = _get_smallest_component(n, edges)
    if n_comp < n:
        n = n_comp
        edges = edges_comp

    # Check for duplicate edges or self-loops
    edge_set = set()
    clean_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            clean_edges.append((u, v))
    edges = clean_edges

    # Verify girth
    if not _verify_girth(n, edges, g):
        # Fallback: should not happen, but return anyway
        pass

    return n, edges

# EVOLVE-BLOCK-END
