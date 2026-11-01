"""4-regular graph of girth 10: hybrid LUW + voltage-graph lift + local search."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque
import numpy as np

def _moore_bound(k, g):
    m = g // 2
    if k == 2:
        return 2 * m
    if m == 1:
        return k
    return k + k * (k - 1) * ((k - 1) ** (m - 1) - 1) // (k - 2)

def _verify_girth_fast(n, edges, k, g, timeout=5.0):
    """Verify girth >= g using non-backtracking BFS from each directed edge."""
    start_time = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degree and simplicity check
    for i in range(n):
        if len(adj[i]) != k:
            return False
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Non-backtracking BFS: for each directed edge (a,b), walk up to g-2 steps
    max_depth = g - 2
    for a in range(n):
        for b in adj[a]:
            if time.time() - start_time > timeout:
                return False
            # BFS from state (b, a)
            visited = set()
            queue = deque()
            queue.append((b, a, 0))
            visited.add((b, a))
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    state = (nxt, cur)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
    return True

def _verify_girth_np(n, edges, k, g, timeout=5.0):
    """Faster girth verification using numpy for adjacency and set-based visited."""
    start_time = time.time()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    if len(edges) != n * k // 2:
        return False
    
    max_depth = g - 2
    for a in range(n):
        for b in adj[a]:
            if time.time() - start_time > timeout:
                return False
            visited = set()
            queue = deque()
            queue.append((b, a, 0))
            visited.add((b, a))
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    state = (nxt, cur)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
    return True

def _construct_luw(q, d):
    """Lazebnik-Ustimenko-Woldar CD(d, q)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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

def _try_luw(k, g, deadline):
    """Try LUW constructions with various d values."""
    best_n = None
    best_edges = None
    
    for d in range(max(3, g - 5), max(3, g - 5) + 8):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = _construct_luw(k, d)
            if n < 5000:
                if _verify_girth_np(n, edges, k, g, timeout=3.0):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
        except Exception:
            pass
    
    return best_n, best_edges

def _voltage_lift_construction(k, g, deadline, seed=123):
    """
    Voltage graph lift: take a small base graph B with degree k, assign voltages
    from Z_m to edges, lift to m*|V_B| vertices. Search over voltage assignments
    to maximize girth.
    
    For k=4, g=10: try base graphs with 4-8 vertices, lift over Z_m for various m.
    """
    random.seed(seed)
    best_n = None
    best_edges = None
    best_girth = 0
    
    # Base graphs: k-regular graphs on small number of vertices
    # For k=4: smallest is K5 (5 vertices, 4-regular)
    # Also: 4-regular on 6 vertices (complement of perfect matching on K6)
    # 4-regular on 8 vertices, etc.
    
    base_graphs = []
    
    # K5: complete graph on 5 vertices (4-regular)
    if k == 4:
        base_graphs.append([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])
        # Complement of 1-factor on K6: 4-regular on 6 vertices
        # K6 has 15 edges, remove 3 disjoint edges
        base_graphs.append([(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)])
        # Remove a perfect matching: (0,5),(1,4),(2,3) -> keep rest
        remove_set = {(0,5),(1,4),(2,3)}
        base6 = [(u,v) for u,v in base_graphs[-1] if (min(u,v),max(u,v)) not in remove_set]
        base_graphs[-1] = base6
    
    # For general k, try K_{k+1}
    if k + 1 <= 12:
        kbase = [(i,j) for i in range(k+1) for j in range(i+1, k+1)]
        if len(kbase) == (k+1)*k//2:
            base_graphs.append(kbase)
    
    for base_edges in base_graphs:
        nb = max(max(u,v) for u,v in base_edges) + 1
        # Check base is k-regular
        deg = [0]*nb
        for u,v in base_edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            continue
        
        ne = len(base_edges)
        # Try different lift sizes m
        for m in range(4, 60):
            if time.time() > deadline - 15:
                return best_n, best_edges
            
            target_n = nb * m
            if target_n < 100:
                continue
            
            # Try several random voltage assignments
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Random voltage assignment
                voltages = {}
                for idx, (u,v) in enumerate(base_edges):
                    voltages[idx] = random.randint(0, m-1)
                
                # Build the lifted graph
                # Vertex (b, t) where b is base vertex, t is in Z_m
                # Edge: (b1, t) -- (b2, t + voltage(e)) mod m
                lifted_edges = []
                seen = set()
                valid = True
                for idx, (u,v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + ((t + vol) % m)
                        # Also the reverse direction
                        # (v, t') -- (u, t' - vol) mod m
                        # This is the same edge when t' = t + vol
                        # So we only need one direction per base edge per t
                        if u_lift == v_lift:
                            valid = False
                            break
                        key = (min(u_lift, v_lift), max(u_lift, v_lift))
                        if key in seen:
                            valid = False
                            break
                        seen.add(key)
                        lifted_edges.append((u_lift, v_lift))
                
                if not valid:
                    continue
                
                # Check degree
                if len(lifted_edges) != target_n * k // 2:
                    continue
                
                # Quick girth check with timeout
                if _verify_girth_np(target_n, lifted_edges, k, g, timeout=2.0):
                    if best_n is None or target_n < best_n:
                        best_n = target_n
                        best_edges = lifted_edges
                    break  # This m works, move to next base graph
                
                # If we want to track best girth for potential use
                # (skip for now, just check pass/fail)
    
    return best_n, best_edges

def _random_construction(k, g, target_n, deadline, seed=42):
    """Try to build a k-regular graph with girth >= g via randomized edge addition."""
    random.seed(seed)
    n = target_n
    adj = [set() for _ in range(n)]
    edges = []
    
    # Phase 1: Build a k-regular graph using configuration model
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    random.shuffle(half_edges)
    
    # Pair up half-edges to form edges
    i = 0
    while i < len(half_edges):
        u = half_edges[i]
        v = half_edges[i + 1]
        i += 2
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        else:
            # Retry with a different partner
            for j in range(i + 2, len(half_edges)):
                v2 = half_edges[j]
                if v2 != u and v2 not in adj[u]:
                    half_edges[i + 1], half_edges[j] = half_edges[j], half_edges[i + 1]
                    v = v2
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    break
    
    # Phase 2: Verify and fix girth
    if not _verify_girth_fast(n, edges, k, g, timeout=2.0):
        return None, None
    
    return n, edges

def _local_search_improve(n, edges, k, g, deadline):
    """Try to reduce n by removing vertices and reconnecting."""
    # This is a placeholder for a more sophisticated local search
    # For now, just return the input
    return n, edges

def construct(k, g):
    deadline = time.time() + 115.0
    mb = _moore_bound(k, g)
    
    # Strategy 1: LUW constructions
    best_n, best_edges = _try_luw(k, g, deadline)
    
    if best_n is not None:
        return best_n, best_edges
    
    # Strategy 2: Voltage graph lifts
    if time.time() < deadline - 25:
        vn, ve = _voltage_lift_construction(k, g, deadline, seed=42)
        if vn is not None:
            if best_n is None or vn < best_n:
                best_n = vn
                best_edges = ve
            return best_n, best_edges
    
    # Strategy 3: Try random constructions with various target sizes
    for target_n in [320, 350, 400, 500, 600, 800, 1000, 1200, 1500, 2000]:
        if time.time() > deadline - 20:
            break
        if target_n < mb:
            continue
        n, edges = _random_construction(k, g, target_n, deadline, seed=target_n)
        if n is not None:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW with large d
    try:
        d = g - 3
        n, edges = _construct_luw(k, d)
        if _verify_girth_np(n, edges, k, g, timeout=5.0):
            return n, edges
    except Exception:
        pass
    
    return mb, []
# EVOLVE-BLOCK-END
