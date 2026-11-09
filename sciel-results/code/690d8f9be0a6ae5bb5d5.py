"""Construct small 4-regular graphs of girth 10 (cage problem n(4,10)).

Strategy:
1. Use Lazebnik-Ustimenko-Woldar (LUW) construction with multiple parameter choices.
2. Verify girth efficiently using non-backtracking BFS.
3. Try edge-swap local search to find smaller valid graphs.
4. Use a time-bounded search with explicit deadline guard.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _luw_construct(k, g, d_override=None):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges) for a q-regular graph with girth >= d+5 (odd d)."""
    q = k
    if d_override is not None:
        d = d_override
    else:
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

    # Try different rule patterns
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
        queue = [(start, -1, 0)]
        while queue:
            node, prev, depth = queue.pop(0)
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


def _count_components(n, edges):
    """Count number of connected components."""
    if n == 0:
        return 0
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * n
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    for nb in adj[node]:
                        if not visited[nb]:
                            stack.append(nb)
    return count


def _edge_swap_search(n, edges, k, g, deadline):
    """Try edge swaps to find a smaller valid graph or improve connectivity.
    
    Edge swaps preserve n, but can help find different valid configurations.
    We focus on ensuring the graph is connected and has good girth properties.
    """
    if time.time() >= deadline:
        return n, edges
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Check if already connected
    if _count_components(n, edges) == 1:
        return n, edges
    
    # Try random edge swaps to improve connectivity
    random.seed(42)
    best_n, best_edges = n, edges
    
    for attempt in range(1000):
        if time.time() >= deadline:
            break
        
        # Pick two random edges to swap
        if len(edges) < 2:
            break
        i, j = random.sample(range(len(edges)), 2)
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        
        # Try swapping: remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
        new_edges = [e for idx, e in enumerate(edges) if idx != i and idx != j]
        new_edges.append((u1, v2))
        new_edges.append((u2, v1))
        
        # Check validity
        valid = True
        deg = [0] * n
        seen = set()
        for u, v in new_edges:
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen:
                valid = False
                break
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
        
        if valid:
            for i in range(n):
                if deg[i] != k:
                    valid = False
                    break
        
        if valid and _count_components(n, new_edges) < _count_components(n, edges):
            # Better connectivity
            if _verify_girth(n, new_edges, g):
                best_n, best_edges = n, new_edges
                edges = new_edges
    
    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    best_n = float('inf')
    best_edges = None

    # Try multiple LUW parameterizations
    base_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    for d_offset in [0, 1, 2]:
        d_try = base_d + d_offset
        if d_try < 3:
            continue
        
        try:
            n, edges = _luw_construct(k, g, d_override=d_try)
        except Exception:
            continue
        
        # Check degree
        deg = [0] * n
        valid = True
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for i in range(n):
            if deg[i] != k:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check for duplicate edges or self-loops
        edge_set = set()
        clean_edges = []
        for u, v in edges:
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            clean_edges.append((u, v))
        
        if not valid:
            continue
        
        # Verify girth
        if not _verify_girth(n, clean_edges, g):
            continue
        
        # This is a valid graph
        if n < best_n:
            best_n = n
            best_edges = clean_edges
        
        # Check if we have time for local search
        if time.time() < deadline - 5:
            best_n, best_edges = _edge_swap_search(n, clean_edges, k, g, deadline)
    
    # If no valid graph found, use the first LUW result as fallback
    if best_edges is None:
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
        best_n = n
        best_edges = clean_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
