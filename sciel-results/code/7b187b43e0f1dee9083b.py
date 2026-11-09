"""Construct a 3-regular graph with girth 16.

Strategy:
1. Build a random 3-regular graph incrementally using constraint satisfaction.
2. Use non-backtracking walks to enforce girth >= g during construction.
3. Fall back to LUW if the random search fails.
"""
import time
import random
from collections import deque

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g."""
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k != 3 or g != 16:
        return _luw_fallback(k, g)
    
    # Try random construction first
    result = _random_construct(3, 16, deadline)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_fallback(k, g)


def _random_construct(k, g, deadline):
    """Random greedy construction with constraint checking."""
    t0 = time.time()
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5.0:
            return None
        
        # Start with a path of length g-1 (to ensure minimum cycle length)
        # Actually, let's build from scratch using a smarter approach.
        
        # Use a tree-based approach: build a random tree of depth ~g/2,
        # then connect leaves with constraints.
        
        depth = g // 2  # 8
        
        # Build a random 3-regular tree skeleton
        # Root has 3 children, each internal non-root has 2 children
        nodes = []
        children = [[] for _ in range(1)]  # start with root
        parent = [-1]
        depth_arr = [0]
        
        def add_node(d, p):
            idx = len(nodes)
            nodes.append((d, p))
            parent.append(p)
            depth_arr.append(d)
            children.append([])
            return idx
        
        root = 0
        frontier = [root]
        
        for d in range(1, depth + 1):
            next_frontier = []
            for u in frontier:
                if d == depth:
                    continue
                n_children = 2 if d > 1 else 3
                for _ in range(n_children):
                    v = add_node(d, u)
                    children[u].append(v)
                    next_frontier.append(v)
            frontier = next_frontier
        
        n_internal = len(nodes)
        leaves = [i for i in range(n_internal) if not children[i]]
        n_leaves = len(leaves)
        
        # Compute LCA for each pair of leaves
        # For efficiency, precompute ancestor sets
        leaf_ancestors = {}
        for lf in leaves:
            anc = set()
            u = lf
            while u >= 0:
                anc.add(u)
                u = parent[u]
            leaf_ancestors[lf] = anc
        
        def lca_depth(u, v):
            # Find deepest common ancestor
            au = leaf_ancestors[u]
            av = leaf_ancestors[v]
            common = au & av
            if not common:
                return 0
            # Find the one with max depth
            max_d = -1
            for node in common:
                if depth_arr[node] > max_d:
                    max_d = depth_arr[node]
            return max_d
        
        # We need: for any matched pair (u,v), dist(u,v) >= g-2 = 14
        # dist = depth[u] + depth[v] - 2*depth[lca] = 2*depth - 2*depth[lca]
        # So 2*depth - 2*depth[lca] >= 14 => depth[lca] <= depth - 7 = 1
        # LCA must be at depth 0 or 1.
        
        # Build allowed matching edges
        leaf_to_idx = {lf: i for i, lf in enumerate(leaves)}
        allowed_adj = [[] for _ in range(n_leaves)]
        
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                u, v = leaves[i], leaves[j]
                if lca_depth(u, v) <= 1:
                    allowed_adj[i].append(j)
                    allowed_adj[j].append(i)
        
        # Find perfect matching in allowed_adj using randomized greedy
        match = _find_random_perfect_matching(allowed_adj, n_leaves)
        
        if match is None:
            continue
        
        # Build edges
        edges = []
        for u in range(n_internal):
            for v in children[u]:
                edges.append((u, v))
        
        for i in range(n_leaves):
            if i < match[i]:
                edges.append((leaves[i], leaves[match[i]]))
        
        n = n_internal
        adj = [[] for _ in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Check degree
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        # Check girth using non-backtracking BFS
        if _check_girth(adj, n, g):
            return n, edges
        
        # If girth check fails, try again
        continue
    
    return None


def _find_random_perfect_matching(adj, n):
    """Find a perfect matching using randomized DFS with restarts."""
    for restart in range(20):
        match = [-1] * n
        
        def dfs(u, visited):
            neighbors = adj[u][:]
            random.shuffle(neighbors)
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    if match[v] == -1 or dfs(match[v], visited):
                        match[v] = u
                        return True
            return False
        
        # Random order for vertices
        order = list(range(n))
        random.shuffle(order)
        
        for u in order:
            visited = [False] * n
            if not dfs(u, visited):
                break
        
        # Check if perfect
        if all(match[i] != -1 for i in range(n)):
            return match
    
    return None


def _check_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for start in range(n):
        # Non-backtracking BFS: track (node, prev) pairs
        # dist[(u,v)] = shortest non-backtracking walk from start to u via v
        dist = {}
        q = deque()
        
        for v in adj[start]:
            dist[(start, v)] = 1
            q.append((start, v))
        
        while q:
            u, prev = q.popleft()
            d = dist[(u, prev)]
            if d >= g - 1:
                continue  # No need to explore further
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start:
                    # Found a cycle of length d + 1
                    if d + 1 < g:
                        return False
                elif (u, w) not in dist:
                    dist[(u, w)] = d + 1
                    q.append((u, w))
    
    return True


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
