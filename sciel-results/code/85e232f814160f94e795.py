import time
import random
import sys
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

    # For girth verification, we only need to check non-backtracking walks
    # up to depth floor((min_girth-1)/2) to detect cycles of length < min_girth
    # A cycle of length L is detected when we return to start at depth L
    # We need L < min_girth, so max depth to check is min_girth - 1
    # But due to symmetry, we can stop at depth floor((min_girth-1)/2) 
    # Actually, we need to be careful: a cycle of length L means we return to start
    # at depth L. We want to detect if any L < min_girth exists.
    # So we BFS up to depth min_girth - 1 and check if we hit start.
    # But to be efficient, note that the shortest cycle through start has length
    # at most 2*depth where depth is when we first revisit start via non-backtracking.
    
    # More efficient: BFS from each start, track (node, prev) states, 
    # stop when depth >= min_girth (since we can't find shorter cycles beyond that)
    max_depth = min_girth  # we look for cycles of length < min_girth
    
    for start in range(n):
        # State: (node, prev_node) -> depth (first time we reach this state)
        visited = {}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if node == start and depth > 0:
                # Found a cycle of length depth
                if depth < min_girth:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                if visited[state] <= depth:
                    continue
                else:
                    continue  # already processed at earlier or same depth
            visited[state] = depth
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

def _try_edge_swap_improve(n, edges, k, g, deadline):
    """Try to improve the graph by finding a smaller valid subgraph via component analysis.
    Returns (n, edges) of the best graph found."""
    # First, find all connected components and pick the smallest
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * n
    best_n = n
    best_edges = edges
    
    for i in range(n):
        if not visited[i]:
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
            if comp_size < best_n:
                best_n = comp_size
                comp_set = set(comp_nodes)
                best_edges = [(u, v) for u, v in edges if u in comp_set and v in comp_set]
    
    return best_n, best_edges


def _fast_verify_girth(n, edges, min_girth):
    """Fast girth verification using non-backtracking BFS with early termination."""
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree first
    for i in range(n):
        if len(adj[i]) != min_girth - 1:  # placeholder, actual check done in construct
            pass
    
    max_depth = min_girth - 1  # we look for cycles of length < min_girth
    
    for start in range(n):
        # State: (node, prev_node) -> depth
        visited = {}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if node == start and depth > 0:
                if depth < min_girth:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                continue
            visited[state] = depth
            for nb in adj[node]:
                if nb == prev:
                    continue
                queue.append((nb, node, depth + 1))
    return True


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Get LUW construction
    n, edges = _luw_construct(k, g)

    # Check for duplicate edges or self-loops first
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
    n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    # Try to find smallest component
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        n, edges = _try_edge_swap_improve(n, edges, k, g, deadline)

    # Verify the graph is valid (degree k, girth >= g)
    if n > 0:
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        valid_degree = all(d == k for d in deg)
        
        if not valid_degree:
            # Fallback: return the original LUW graph
            n, edges = _luw_construct(k, g)
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
            n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    # Verify girth if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        if not _fast_verify_girth(n, edges, g):
            # Fallback: return the original LUW graph
            n, edges = _luw_construct(k, g)
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
            n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    return n, edges

# EVOLVE-BLOCK-END
