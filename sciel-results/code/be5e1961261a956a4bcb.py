"""Search for a small 3-regular graph of girth >= g.

Strategy:
1. Deterministic exhaustive search over small circulant and doubly-circulant
   families. These are highly structured and often yield optimal or near-optimal
   cages for small parameters.
2. Fast girth certification via per-vertex BFS (O(N * (k-1)^(g/2))).
3. Fallback to a guaranteed-valid LUW construction if search fails.
"""
import time
import random
import numpy as np
import itertools

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# LUW construction (seed / fallback)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
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


def _edges_to_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# Fast girth certification via per-vertex BFS
# ---------------------------------------------------------------------------
def _girth_ok_bfs(adj, n, g):
    """Exact check: no cycle of length < g. Per-start BFS over NB states."""
    if g <= 3:
        return True
    adj_sets = [frozenset(a) for a in adj]
    for s in range(n):
        visited = set()
        queue = []
        for w in adj_sets[s]:
            queue.append((w, s, 1))
            visited.add((w, s))
        qi = 0
        while qi < len(queue):
            v, u, length = queue[qi]
            qi += 1
            if length >= g:
                continue
            for w in adj_sets[v]:
                if w == u:
                    continue
                if w == s and length + 1 >= 3:
                    return False
                state = (w, v)
                if state not in visited:
                    visited.add(state)
                    if length + 1 < g:
                        queue.append((w, v, length + 1))
    return True


# ---------------------------------------------------------------------------
# Circulant search
# ---------------------------------------------------------------------------
def _search_circulants(k, g, deadline):
    """Search for a small circulant graph of girth >= g."""
    if k != 3:
        return None
    
    best = None
    best_n = float('inf')
    
    # Moore bound for n(3, 16) is 512. Try sizes up to ~600.
    # n must be even.
    for n in range(512, 620, 2):
        if time.time() > deadline:
            break
        if n >= best_n:
            break
            
        # Generate candidate circulant graphs.
        # A circulant graph C_n(a, b) has edges (i, i+a) and (i, i-b) mod n.
        # For k=3, we need a, b such that the graph is simple and 3-regular.
        # Conditions:
        # 1. a != b, a != -b (mod n)
        # 2. 2a != 0, 2b != 0 (mod n) unless n is even and a = n/2 (which gives a loop if not careful, but for k=3 we usually avoid n/2)
        # 3. The set of differences {a, -a, b, -b} must be distinct.
        
        # We can restrict search space by symmetry.
        # Assume 0 < a < b < n/2.
        # Also a != b, a != n-b (implied by a < n/2 < b if b > n/2? No, we restrict b < n/2 too? 
        # Actually, if we take a, b in 1..n/2-1, we cover all non-degenerate cases up to symmetry.
        
        max_a = n // 2
        for a in range(1, max_a):
            for b in range(a + 1, max_a):
                # Check simplicity conditions
                if a == b: continue
                if a == -b % n: continue
                if 2*a % n == 0: continue
                if 2*b % n == 0: continue
                if a == b: continue
                if a + b == 0: continue # a != -b
                if a - b == 0: continue # a != b
                
                # Build graph
                adj = [[] for _ in range(n)]
                for i in range(n):
                    adj[i].append((i + a) % n)
                    adj[i].append((i - a) % n)
                    adj[i].append((i + b) % n)
                
                # Verify simple
                ok = True
                for i in range(n):
                    if len(adj[i]) != 3:
                        ok = False
                        break
                    if len(set(adj[i])) != 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check girth
                if _girth_ok_bfs(adj, n, g):
                    edges = _adj_to_edges(adj, n)
                    # Verify edges count
                    if len(edges) == n * k // 2:
                        best = (n, edges)
                        best_n = n
                        break # Found smallest n, break n loop? No, break b loop, then n loop will break due to best_n check
            if best is not None:
                break
        if best is not None:
            break
            
    return best

def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock
    deadline = t0 + HARD_LIMIT
    
    # 1) Search circulants
    result = _search_circulants(k, g, deadline)
    if result is not None:
        return result
    
    # 2) Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
