import math
import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 120  # 20s margin
    
    # Phase 1: Cyclic Cayley Graphs (Fast Pruning)
    moore_bound = 243
    for m in range(moore_bound, 450):
        if time.time() > deadline:
            break
        found = _search_cyclic_fast(m, g)
        if found is not None:
            a, b = found
            edges = _build_cyclic_edges(m, a, b)
            return m, edges

    # Phase 2: Random 4-regular Graphs
    # Try to find a random graph with girth >= g
    # Start from Moore bound, go up
    for n in range(moore_bound, 500):
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        # Number of attempts depends on time remaining
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = _generate_random_4_reg(n)
            if edges is None:
                continue
            if _check_girth_bfs(n, edges, g):
                return n, edges

    # Phase 3: Fallback LUW
    return _construct_fallback(4, g)

def _search_cyclic_fast(m, g):
    # Prune (a,b) based on small cycles
    for a in range(1, m // 2 + 1):
        if 2 * a == m:
            continue
        for b in range(a + 1, m // 2 + 1):
            if 2 * b == m:
                continue
            if a + b == m or b - a == 0:
                continue
            if math.gcd(math.gcd(a, b), m) != 1:
                continue
            
            # Fast check for L=3,4,5
            if _has_small_cycle_cyclic(m, a, b, 3):
                continue
            if _has_small_cycle_cyclic(m, a, b, 4):
                continue
            if _has_small_cycle_cyclic(m, a, b, 5):
                continue
            
            # Full check
            if _check_girth_cyclic(m, a, b, g):
                return (a, b)
    return None

def _has_small_cycle_cyclic(m, a, b, L):
    # Check for non-backtracking closed walks of length L
    # Using the sum property
    S = [a, -a, b, -b]
    # For L=3:
    if L == 3:
        # a+a+a, a+a-b, a-b+b (invalid), a-b-a (invalid), etc.
        # Valid non-backtracking sequences of 3:
        # x, y, z. y != -x, z != -y.
        # x+y+z = 0
        # Possible sums:
        # 3a, 3b, 2a-b, 2b-a, a-b-c (no c)
        # Check 3a % m == 0
        if 3 * a % m == 0: return True
        if 3 * b % m == 0: return True
        if (2 * a - b) % m == 0: return True
        if (2 * b - a) % m == 0: return True
        # Also a - b - a is backtrack.
        # a + b - a is backtrack.
        # a - b + b is backtrack.
        # So only 3a, 3b, 2a-b, 2b-a.
        return False
        
    # For L=4:
    if L == 4:
        # x,y,z,w. y!=-x, z!=-y, w!=-z.
        # Sum = 0.
        # 4a, 4b, 2a+2b, 2a-2b, a+b+a-b (backtrack at end? a+b-a+b = 2b, w=b, z=-a, y=b, x=a. y!=-x (b!=-a), z!=-y (-a!=-b), w!=-z (b!=-(-a)=a). OK.)
        # So a+b-a+b = 2b. If 2b % m == 0, cycle.
        # Similarly a-b+a-b = -2b.
        # a+a-b-b = 2a-2b.
        # Check 4a, 4b, 2a+2b, 2a-2b, 2b, -2b.
        if 4 * a % m == 0: return True
        if 4 * b % m == 0: return True
        if (2 * a + 2 * b) % m == 0: return True
        if (2 * a - 2 * b) % m == 0: return True
        if 2 * b % m == 0: return True
        if -2 * b % m == 0: return True
        if 2 * a % m == 0: return True
        return False
        
    # For L=5:
    # Too many cases, use DFS for L=5 if needed, or just let full check handle it.
    # But L=5 is 4*3^4 = 324 cases. Fast enough.
    if L == 5:
        return _nb_walk_cyclic(m, [a, -a, b, -b], [2, 3, 0, 1], 5)
        
    return False

def _check_girth_cyclic(m, a, b, g):
    S = [a, -a, b, -b]
    inv = [2, 3, 0, 1]
    for L in range(6, g): # 3,4,5 already checked
        if _nb_walk_cyclic(m, S, inv, L):
            return False
    return True

def _nb_walk_cyclic(m, S, inv, L):
    def _dfs(depth, s, last, first):
        if depth == L:
            return s % m == 0 and inv[last] != first
        for ng in range(4):
            if ng != inv[last]:
                if _dfs(depth + 1, s + S[ng], ng, first):
                    return True
        return False
    for sg in range(4):
        if _dfs(1, S[sg], sg, sg):
            return True
    return False

def _build_cyclic_edges(m, a, b):
    edge_set = set()
    for i in range(m):
        for s in [a, b]:
            j = (i + s) % m
            edge_set.add((min(i, j), max(i, j)))
    return sorted(edge_set)

def _generate_random_4_reg(n):
    # Configuration model
    half_edges = []
    for i in range(n):
        for _ in range(4):
            half_edges.append(i)
    random.shuffle(half_edges)
    
    edges = set()
    for i in range(0, 2 * n, 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            return None # Loop
        if (min(u, v), max(u, v)) in edges:
            return None # Multi-edge
        edges.add((min(u, v), max(u, v)))
        
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 4 for d in deg):
        return None
        
    return list(edges)

def _check_girth_bfs(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles < g
    # BFS from each node
    for start in range(n):
        # State: (node, prev_node)
        # dist: dict (node, prev) -> dist
        # We want to find a cycle.
        # If we reach a node (u, v) that was visited at dist d1, and now at d2,
        # and the path is simple?
        # Simpler: Just do BFS and track parent.
        # If we find a back edge to an ancestor, calculate cycle length.
        # But we need to ensure no backtracking in the walk?
        # Girth is the length of the shortest cycle in the graph.
        # A cycle is a simple path that closes.
        # BFS from start.
        # If we encounter a node already visited, we found a cycle.
        # But we need the SHORTEST one.
        # Standard BFS finds shortest path to each node.
        # If we have a cycle, it will be detected.
        # To find girth, we can do BFS from each node and find the shortest cycle involving that node?
        # Actually, the shortest cycle in the graph will be found by BFS from any node on that cycle.
        # So:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        min_cycle = float('inf')
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # But this might not be the shortest cycle in the graph,
                    # just a cycle.
                    # However, BFS explores in order of distance.
                    # The first cycle found might not be the global minimum.
                    # But if we run BFS from ALL nodes, we will find the global minimum.
                    # Wait, if we run BFS from start, and find a cycle of length L,
                    # is it the shortest?
                    # Yes, because BFS finds shortest paths.
                    # The cycle length is dist[u] + dist[v] + 1.
                    # Since BFS expands layers, the first time we see a cross edge,
                    # it gives the shortest cycle containing that edge?
                    # Actually, we should track the minimum cycle length found.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return False
                        
        # If no cycle found, graph is a tree? No, 4-regular graph has cycles.
        # If min_cycle is inf, no cycle? Impossible for 4-regular with n>1.
        # But if we return True, it means no cycle < g was found.
        # We must ensure we checked all nodes.
        
    return True

def _construct_fallback(k, g):
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
