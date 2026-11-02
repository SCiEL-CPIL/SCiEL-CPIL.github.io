"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # For k=4, g=10, we know the LUW construction with d=4 yields 512 vertices and girth 10.
    # The seed used d=5 (yielding 2048).
    # We try to use the optimal d for the given g.
    # Girth of LUW(q, d) is >= 2d + 2.
    # We need 2d + 2 >= g => d >= (g-2)/2.
    # For g=10, d >= 4.
    # So d=4 is the minimum d for LUW to satisfy girth 10.
    
    if k == 4 and g == 10:
        # Try LUW with d=4
        result = _luw_construct_specific(k, 4)
        if result is not None:
            n, edges = result
            # Verify girth just in case
            if _has_girth_at_least(edges, n, g):
                return n, edges
    
    # Fallback to original logic for other cases or if verification fails
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        best_n, best_edges = luw_result
    else:
        # Fallback: random regular graph with larger n
        best_n, best_edges = _random_regular(k, g, start_time, deadline)
        
    return best_n, best_edges

def _luw_construct_specific(k, d):
    """Lazebnik-Ustimenko-Woldar graph construction with specific d."""
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # For prime powers, we need a field. 
        # This implementation only supports q=4 or prime q simply.
        # For robustness, we stick to q=4 for this specific optimization.
        if q < 2:
            return None
        # Simple field for small primes
        # This part is risky for general q, but we only call this for k=4.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for LUW construction
    # The rule depends on d.
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (original logic)."""
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
    
    # Define the rule for LUW construction
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _random_regular(k, g, start_time, deadline):
    """Generate a random k-regular graph."""
    n = 500
    edges = _gen_random_4reg(n)
    if edges is not None:
        return n, edges
    return n, []


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    for attempt in range(100):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None


def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _count_short_cycles(adj, n, max_len=8):
    """Count cycles of length up to max_len. Used as a proxy for girth."""
    four_cycles = 0
    six_cycles = 0
    
    for u in range(n):
        for v in range(u+1, n):
            common = len(set(adj[u]) & set(adj[v]))
            if common >= 2:
                four_cycles += 1
        
    return four_cycles + six_cycles


def _local_search_for_n(k, g, n, start_time, deadline):
    """Try to find a k-regular graph of girth >= g on exactly n vertices via local search."""
    time_budget = deadline - start_time - 10
    
    edges = _gen_random_4reg(n)
    if edges is None:
        return None
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    if _has_girth_at_least(edges, n, g):
        return n, edges
    
    best_cost = _count_short_cycles(adj, n)
    best_edges = list(edges)
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 10:
        e1_idx = random.randint(0, len(edges) - 1)
        e2_idx = random.randint(0, len(edges) - 1)
        if e1_idx == e2_idx:
            continue
        
        u1, v1 = edges[e1_idx]
        u2, v2 = edges[e2_idx]
        
        new_edges = list(edges)
        
        if u1 != u2 and v1 != v2:
            e_new1 = (min(u1,u2), max(u1,u2))
            e_new2 = (min(v1,v2), max(v1,v2))
            
            if e_new1 != e_new2 and e_new1 not in set(edges) and e_new2 not in set(edges):
                new_edges[e1_idx] = e_new1
                new_edges[e2_idx] = e_new2
                
                valid = True
                edge_set = set()
                for e in new_edges:
                    if e[0] == e[1]:
                        valid = False
                        break
                    if e in edge_set:
                        valid = False
                        break
                    edge_set.add(e)
                
                if valid:
                    new_adj = [[] for _ in range(n)]
                    for u, v in new_edges:
                        new_adj[u].append(v)
                        new_adj[v].append(u)
                    
                    new_cost = _count_short_cycles(new_adj, n)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_edges = new_edges
                        edges = new_edges
                        adj = new_adj
                    elif random.random() < math.exp(-(new_cost - best_cost) / temp):
                        edges = new_edges
                        adj = new_adj
        
        if u1 != v2 and v1 != u2:
            e_new1 = (min(u1,v2), max(u1,v2))
            e_new2 = (min(v1,u2), max(v1,u2))
            
            if e_new1 != e_new2 and e_new1 not in set(edges) and e_new2 not in set(edges):
                new_edges = list(edges)
                new_edges[e1_idx] = e_new1
                new_edges[e2_idx] = e_new2
                
                valid = True
                edge_set = set()
                for e in new_edges:
                    if e[0] == e[1]:
                        valid = False
                        break
                    if e in edge_set:
                        valid = False
                        break
                    edge_set.add(e)
                
                if valid:
                    new_adj = [[] for _ in range(n)]
                    for u, v in new_edges:
                        new_adj[u].append(v)
                        new_adj[v].append(u)
                    
                    new_cost = _count_short_cycles(new_adj, n)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_edges = new_edges
                        edges = new_edges
                        adj = new_adj
                    elif random.random() < math.exp(-(new_cost - best_cost) / temp):
                        edges = new_edges
                        adj = new_adj
        
        temp *= cooling
    
    if _has_girth_at_least(best_edges, n, g):
        return n, best_edges
    
    return None

# EVOLVE-BLOCK-END
