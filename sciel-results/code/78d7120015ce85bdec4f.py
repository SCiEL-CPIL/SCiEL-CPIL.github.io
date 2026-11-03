import numpy as np
import random
import time
from collections import deque
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses voltage lift technique with a base graph from PG(2,3).
    """
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _fallback_luw(k, g)
    
    # Deadline guard
    deadline = time.time() + 120.0
    
    # Build PG(2,3) incidence graph as base
    # Points in PG(2,3): 13 points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: first non-zero coordinate is 1
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z * inv_y) % 3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    # Lines in PG(2,3): 13 lines
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Canonicalize normal vector
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c * inv_b) % 3)
                else:
                    n = (0, 0, 1)
                
                if n in lines:
                    continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph adjacency
    n_base = 26  # 13 points + 13 lines
    base_adj = [[] for _ in range(n_base)]
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_adj[i].append(13 + j)
                base_adj[13 + j].append(i)
                base_edges.append((i, 13 + j))
    
    # Use voltage lift over Z_m
    # Moore bound for (4,10) is 682, so we need n_base * m >= 682
    # 26 * m >= 682 => m >= 27
    # Try m = 28, 30, 32, etc.
    
    best_result = None
    for m in [28, 30, 32, 36, 40]:
        if time.time() > deadline - 10:
            break
        n_vertices = n_base * m
        if n_vertices < 682:
            continue
        
        # Try to find valid voltages
        result = _find_voltages_efficient(base_adj, base_edges, m, g, deadline)
        if result is not None:
            lift_edges = result
            # Verify girth
            if _verify_girth(n_vertices, lift_edges, g):
                best_result = (n_vertices, lift_edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return _fallback_luw(k, g)


def _find_voltages_efficient(base_adj, base_edges, m, g, deadline):
    """
    Find voltage assignments such that no cycle of length < g in the lift has total voltage 0 mod m.
    Uses a smarter approach: enumerate short cycles first, then use constraint satisfaction.
    """
    n_base = len(base_adj)
    
    # Precompute all short cycles (length 6 and 8) in the base graph using BFS-based enumeration
    cycles_6 = _find_cycles_of_length_fast(base_adj, 6)
    cycles_8 = _find_cycles_of_length_fast(base_adj, 8)
    
    n_cycles_6 = len(cycles_6)
    n_cycles_8 = len(cycles_8)
    
    # For each trial, assign random voltages and check
    # Use a more systematic approach: try different random seeds and strategies
    max_trials = 20000
    for trial in range(max_trials):
        if time.time() > deadline - 15:
            break
        
        # Strategy: random with some structure
        voltages = {}
        for u, v in base_edges:
            vol = random.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        
        # Check all 6-cycles
        valid = True
        for cycle in cycles_6:
            s = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check all 8-cycles
        for cycle in cycles_8:
            s = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Build lift edges
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u * m + i, v * m + (i + vu) % m))
        
        return lift_edges
    
    return None


def _find_cycles_of_length_fast(base_adj, length):
    """
    Find all cycles of given length in the base graph using a more efficient approach.
    Returns list of cycles, where each cycle is a list of vertices.
    """
    n = len(base_adj)
    cycles = []
    seen = set()
    
    if length == 6:
        # Find all 6-cycles: v0-v1-v2-v3-v4-v5-v0
        # Use symmetry: fix v0 < v1 < v3 to reduce duplicates
        for v0 in range(n):
            for v1 in base_adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in base_adj[v1]:
                    if v2 == v0:
                        continue
                    for v3 in base_adj[v2]:
                        if v3 == v0 or v3 == v1:
                            continue
                        if v3 < v1:
                            continue
                        for v4 in base_adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2:
                                continue
                            for v5 in base_adj[v4]:
                                if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v3:
                                    continue
                                if v0 in base_adj[v5]:
                                    # Check if this is a new cycle (avoid duplicates)
                                    cycle = [v0, v1, v2, v3, v4, v5]
                                    # Canonicalize: start with smallest vertex, then two directions
                                    min_idx = cycle.index(min(cycle))
                                    canon1 = tuple(cycle[min_idx:] + cycle[:min_idx])
                                    canon2 = tuple([cycle[min_idx]] + list(reversed(cycle[:min_idx])))
                                    canon = min(canon1, canon2)
                                    if canon not in seen:
                                        seen.add(canon)
                                        cycles.append(list(canon))
    
    elif length == 8:
        # Find all 8-cycles: v0-v1-v2-v3-v4-v5-v6-v7-v0
        # Use symmetry: fix v0 < v1 < v4 to reduce duplicates
        for v0 in range(n):
            for v1 in base_adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in base_adj[v1]:
                    if v2 == v0:
                        continue
                    for v3 in base_adj[v2]:
                        if v3 == v0 or v3 == v1:
                            continue
                        for v4 in base_adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2:
                                continue
                            if v4 < v1:
                                continue
                            for v5 in base_adj[v4]:
                                if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v3:
                                    continue
                                for v6 in base_adj[v5]:
                                    if v6 == v0 or v6 == v1 or v6 == v2 or v6 == v3 or v6 == v4:
                                        continue
                                    for v7 in base_adj[v6]:
                                        if v7 == v0 or v7 == v1 or v7 == v2 or v7 == v3 or v7 == v4 or v7 == v5:
                                            continue
                                        if v0 in base_adj[v7]:
                                            # Check if this is a new cycle (avoid duplicates)
                                            cycle = [v0, v1, v2, v3, v4, v5, v6, v7]
                                            # Canonicalize: start with smallest vertex, then two directions
                                            min_idx = cycle.index(min(cycle))
                                            canon1 = tuple(cycle[min_idx:] + cycle[:min_idx])
                                            canon2 = tuple([cycle[min_idx]] + list(reversed(cycle[:min_idx])))
                                            canon = min(canon1, canon2)
                                            if canon not in seen:
                                                seen.add(canon)
                                                cycles.append(list(canon))
    
    return cycles


def _verify_girth(n_vertices, edges, g):
    """
    Verify that the graph has girth at least g using BFS from each vertex.
    """
    adj = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n_vertices):
        dist = [-1] * n_vertices
        parent = [-1] * n_vertices
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    queue.append(nb)
                elif parent[nb] != cur and parent[cur] != nb:
                    cycle_len = dist[nb] + dist[cur] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
    
    return min_girth >= g


def _fallback_luw(k, g):
    """
    Fallback LUW (Lubotzky-Phillips-Sarnak style) construction.
    """
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
