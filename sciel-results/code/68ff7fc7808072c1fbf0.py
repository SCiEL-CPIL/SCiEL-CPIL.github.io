import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Moore bound for lower bound on n
    if g % 2 == 0:
        m = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(m))
    else:
        m = (g - 1) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(m))
        
    best_graph = None
    best_n = float('inf')
    
    # Try LUW construction first as a baseline (fast)
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        best_graph = (n_luw, edges_luw)
        best_n = n_luw
    except Exception:
        pass

    # Now try to find a smaller graph using random + local search
    current_n = max(moore_bound, 20)
    if current_n % 2 != 0:
        current_n += 1  # k=5 is odd, so n must be even for 5-regular graph
        
    attempts = 0
    max_attempts = 30  # Limit attempts to avoid infinite loop
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate random k-regular graph
        try:
            edges = _generate_random_regular_graph(current_n, k)
        except Exception:
            current_n += 10
            continue
            
        # Check girth
        girth = _check_girth(current_n, edges, g)
        
        if girth >= g:
            # Found a valid graph
            if current_n < best_n:
                best_graph = (current_n, edges)
                best_n = current_n
            return best_graph
        else:
            # Girth is too small. Try to increase it via local search.
            if current_n < 600:
                edges = _local_search_girth(current_n, edges, k, g, deadline)
                girth = _check_girth(current_n, edges, g)
                if girth >= g:
                    if current_n < best_n:
                        best_graph = (current_n, edges)
                        best_n = current_n
                        return best_graph
            # Increase n for next attempt
            current_n += 10
            if current_n % 2 != 0:
                current_n += 1

    return best_graph

def _luw_construct(k, g):
    """Fallback LUW construction from seed."""
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
    return len(order), edges

def _generate_random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using edge swapping."""
    if n % 2 != 0:
        raise ValueError("n must be even for odd k")
    
    # Construct base k-regular graph
    # For k=5: C_n(1,2) is 4-regular, add perfect matching for 5-regular
    edge_set = set()
    
    for i in range(n):
        for offset in range(1, k // 2 + 1):  # offsets 1, 2 for k=5
            j = (i + offset) % n
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
    
    # If k is odd, add perfect matching
    if k % 2 == 1:
        half = n // 2
        for i in range(half):
            j = i + half
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
    
    edges_list = list(edge_set)
    
    # Perform random edge swaps to randomize
    swaps = n * 3
    for _ in range(swaps):
        if len(edges_list) < 2:
            break
        i1, i2 = random.randint(0, len(edges_list) - 1), random.randint(0, len(edges_list) - 1)
        if i1 == i2:
            continue
        u1, v1 = edges_list[i1]
        u2, v2 = edges_list[i2]
        
        candidates = [
            [(u1, u2), (v1, v2)],
            [(u1, v2), (v1, u2)]
        ]
        
        for new_edges in candidates:
            e1 = (min(new_edges[0][0], new_edges[0][1]), max(new_edges[0][0], new_edges[0][1]))
            e2 = (min(new_edges[1][0], new_edges[1][1]), max(new_edges[1][0], new_edges[1][1]))
            
            if e1[0] == e1[1] or e2[0] == e2[1]:
                continue
            
            old_keys = {(min(u1, v1), max(u1, v1)), (min(u2, v2), max(u2, v2))}
            if e1 in old_keys or e2 in old_keys:
                continue
            if e1 in edge_set or e2 in edge_set:
                continue
            
            edge_set.discard((min(u1, v1), max(u1, v1)))
            edge_set.discard((min(u2, v2), max(u2, v2)))
            edge_set.add(e1)
            edge_set.add(e2)
            edges_list[i1] = e1
            edges_list[i2] = e2
            break
    
    return edges_list

def _check_girth(n, edges, target_girth):
    """Check if girth is at least target_girth."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    half_target = target_girth // 2
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= half_target:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif dist[v] > dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < target_girth:
                            return min_girth
    return min_girth

def _local_search_girth(n, edges, k, g, deadline):
    """Attempt to increase girth via edge swaps."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Find a short cycle via BFS
        short_cycle_found = False
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif dist[v] > dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Try to break this cycle
                            # Pick edge (u, v) and try to swap
                            u1, v1 = min(u, v), max(u, v)
                            
                            # Find another edge
                            all_edges = list(edge_set)
                            if len(all_edges) < 2:
                                break
                            u2, v2 = random.choice(all_edges)
                            if (u2, v2) == (u1, v1):
                                continue
                            
                            new_e1 = (min(u1, u2), max(u1, u2))
                            new_e2 = (min(v1, v2), max(v1, v2))
                            
                            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                                continue
                            if new_e1 == (u1, v1) or new_e2 == (u2, v2):
                                continue
                            if new_e1 in edge_set or new_e2 in edge_set:
                                continue
                            
                            edge_set.discard((u1, v1))
                            edge_set.discard((u2, v2))
                            edge_set.add(new_e1)
                            edge_set.add(new_e2)
                            
                            adj[u1].remove(v1)
                            adj[v1].remove(u1)
                            adj[u2].remove(v2)
                            adj[v2].remove(u2)
                            
                            adj[u1].add(u2)
                            adj[u2].add(u1)
                            adj[v1].add(v2)
                            adj[v2].add(v1)
                            
                            short_cycle_found = True
                            break
                if short_cycle_found:
                    break
            if short_cycle_found:
                break
        
        if not short_cycle_found:
            break
    
    return list(edge_set)

# EVOLVE-BLOCK-END
