import math
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    Uses a multi-strategy approach:
    1. Try known explicit constructions (Lazebnik-Ustimenko-Woldar)
    2. Try targeted random search near known best for specific (k,g)
    3. Return best valid graph found within time limit
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    best_result = None
    
    # Strategy 1: LUW construction as guaranteed fallback
    if time.time() < deadline - 5:
        try:
            n_luw, edges_luw = _luw_graph(k, g)
            if n_luw > 0:
                best_result = (n_luw, edges_luw)
        except Exception:
            pass
    
    # Strategy 2: Targeted random search for specific (k,g) pairs
    # For k=5, g=9: known best is 1116, search near that
    if k == 5 and g == 9:
        # Search from near known best downward
        search_start = 1110
        search_end = 1000
        step = 10
        
        for n in range(search_start, search_end - 1, -step):
            if time.time() > deadline - 10:
                break
            
            # Try multiple random graphs at this size
            max_attempts = 30
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                try:
                    edges = _random_k_regular_fast(n, k)
                    if edges is None:
                        continue
                    
                    # Fast girth check with early termination
                    if _has_girth_at_least_fast(g, n, edges):
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
                        # Found a valid graph at this size, try smaller
                        break
                except Exception:
                    continue
            
            # If we found a valid graph, continue to smaller sizes
            # If not, keep trying at this size or move on
    
    # Strategy 3: General random search for other (k,g)
    elif time.time() < deadline - 10:
        try:
            n_random, edges_random = _random_regular_with_girth(k, g, max_tries=30, time_limit=20)
            if n_random > 0:
                if best_result is None or n_random < best_result[0]:
                    best_result = (n_random, edges_random)
        except Exception:
            pass
    
    # Fallback: if nothing found, use LUW
    if best_result is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_graph(k, d)
        if n > 0:
            best_result = (n, edges)
        else:
            if k == 1:
                best_result = (2, [(0, 1)])
            elif k == 2:
                best_result = (g, [(i, (i+1) % g) for i in range(g)])
            else:
                best_result = (0, [])
    
    return best_result


def _luw_graph(q, d):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    q-regular bipartite graph with girth at least d+5 (for odd d).
    """
    if q < 3:
        return 0, []
    
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
    
    return len(order), edges


def _random_k_regular_fast(n, k):
    """
    Generate a random k-regular graph on n vertices using configuration model.
    Optimized for speed: uses set for edge tracking.
    Returns list of edges or None if not simple.
    """
    if n * k < 2 or (n * k) % 2 != 0:
        return None
    
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return None
        edge_set.add((u, v))
        edges.append((u, v))
    
    return edges


def _has_girth_at_least_fast(g, n, edges):
    """
    Check if graph has no cycle shorter than g.
    Uses BFS from each vertex with early termination.
    Optimized: stops as soon as a short cycle is found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _random_regular_with_girth(k, g, max_tries=50, time_limit=20):
    """
    Try to construct a k-regular graph with girth >= g by random generation.
    """
    start_time = time.time()
    
    if g % 2 == 0:
        h = g // 2 - 1
        moore = 2 * k * (k - 1) ** h if h >= 0 else 2 * k
    else:
        h = (g - 1) // 2
        moore = 2 * (k - 1) ** h + k * (k - 1) ** (h - 1) if h >= 1 else 2 * k
    
    moore = max(moore, 2 * k)
    
    for n in range(moore, moore + 30):
        if time.time() - start_time > time_limit:
            break
        for attempt in range(max_tries):
            if time.time() - start_time > time_limit:
                break
            try:
                edges = _random_k_regular_fast(n, k)
                if edges is not None and _has_girth_at_least_fast(g, n, edges):
                    return n, edges
            except Exception:
                pass
    
    return 0, []
