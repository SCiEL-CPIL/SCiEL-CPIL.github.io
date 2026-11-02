import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Moore bound for (3, 16): 2 * (3^7 - 1) / (3 - 1) = 512
    # We need a 3-regular graph with girth >= 16 and n >= 512.
    # Try to construct a graph with as few vertices as possible.
    
    # Strategy 1: Try to use the LUW construction (which gives n = 2*q*d vertices)
    # For k=3, g=16, LUW gives n = 2*3*13 = 78 vertices? No, that's not right.
    # Let me re-read the LUW construction.
    
    # Actually, let's just use the LUW construction as a fallback and try to improve.
    
    # The current seed program uses a lift of the Petersen graph.
    # The Petersen graph has 10 vertices. A lift of size M has 10*M vertices.
    # We need 10*M >= 512 => M >= 52.
    # But the girth of a random lift of a graph with girth 5 and lift size M
    # is approximately 5 + 2*log2(M). For M=52, 2*log2(52) ≈ 2*5.7 = 11.4,
    # so girth ≈ 5 + 11.4 = 16.4. This is promising!
    
    # Let's try random lifts of the Petersen graph with M >= 52.
    
    petersen_edges = _get_petersen_edges()
    
    best_n = float('inf')
    best_edges = None
    
    # Try M from 52 upward
    for M in range(52, 200):
        if time.time() > deadline:
            break
            
        n = 10 * M
        if n >= best_n:
            break
        
        # Try several random lifts
        num_attempts = 30
        for _ in range(num_attempts):
            if time.time() > deadline:
                break
            edges = _random_lift(petersen_edges, M)
            if edges is not None and _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
                break
        
        # If we found a graph for this M, we can stop (since larger M gives larger n)
        if best_n == n:
            break
    
    # If no lift worked, fall back to LUW construction
    if best_n == float('inf'):
        n_luw, edges_luw = _construct_luw(k, g)
        return n_luw, edges_luw
        
    return best_n, best_edges

def _get_petersen_edges():
    # Petersen graph: 10 vertices, 15 edges
    # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i+1) % 5))
    # Inner star
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    """
    Create a random M-lift of the base graph.
    """
    n_base = 10
    n_lift = n_base * M
    
    # Assign a random permutation to each edge
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
            
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
            
    return lift_edges

def _has_girth_at_least(n, edges, g):
    """Check if graph has no cycle shorter than g. Uses BFS from each vertex with parent tracking."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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
# EVOLVE-BLOCK-END
