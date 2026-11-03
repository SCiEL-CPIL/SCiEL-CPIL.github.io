import numpy as np
import random
import time
import sys

def construct(k, g):
    deadline = time.time() + 110
    
    if k == 4 and g == 10:
        # Construct PG(2,3) incidence graph (26 vertices: 13 points, 13 lines)
        points = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if x == 0 and y == 0 and z == 0: continue
                    if x != 0:
                        inv_x = 1 if x == 1 else 2
                        p = (1, (y*inv_x)%3, (z*inv_x)%3)
                    elif y != 0:
                        inv_y = 1 if y == 1 else 2
                        p = (0, 1, (z*inv_y)%3)
                    else:
                        p = (0, 0, 1)
                    if p not in points:
                        points.append(p)
        
        lines = []
        line_points = []
        seen_lines = set()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if a == 0 and b == 0 and c == 0: continue
                    if a != 0:
                        inv_a = 1 if a == 1 else 2
                        n = (1, (b*inv_a)%3, (c*inv_a)%3)
                    elif b != 0:
                        inv_b = 1 if b == 1 else 2
                        n = (0, 1, (c*inv_b)%3)
                    else:
                        n = (0, 0, 1)
                    
                    if n in seen_lines: continue
                    seen_lines.add(n)
                    lines.append(n)
                    pts = []
                    for p in points:
                        if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                            pts.append(p)
                    line_points.append(pts)
        
        base_edges = []
        for i, p in enumerate(points):
            for j, lp in enumerate(line_points):
                if p in lp:
                    base_edges.append((i, 13+j))
        
        n_base = 26
        n_edges = len(base_edges)
        
        # Build adjacency for base graph
        adj = [[] for _ in range(n_base)]
        edge_index = {}
        for idx, (u, v) in enumerate(base_edges):
            adj[u].append(v)
            adj[v].append(u)
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
        # Find all non-backtracking closed walks of length 6 and 8
        walks_6 = set()
        walks_8 = set()
        
        def get_walk_edges(walk):
            edges = []
            for i in range(len(walk)-1):
                u, v = walk[i], walk[i+1]
                edges.append(edge_index[(u, v)])
            return tuple(edges)
        
        for start in range(n_base):
            for first in adj[start]:
                def dfs_nb(current, prev, length, path):
                    if length == 0:
                        if current == start:
                            e_tuple = get_walk_edges(path)
                            if len(path) - 1 == 6:
                                walks_6.add(e_tuple)
                            elif len(path) - 1 == 8:
                                walks_8.add(e_tuple)
                        return
                    
                    for nb in adj[current]:
                        if nb == prev:
                            continue
                        dfs_nb(nb, current, length - 1, path + [nb])
                
                dfs_nb(first, start, 5, [start, first])
                dfs_nb(first, start, 7, [start, first])
        
        all_walks = list(walks_6) + list(walks_8)
        n_walks = len(all_walks)
        
        # Precompute constraint matrix: for each walk, which edges it uses (with multiplicity)
        # constraint_matrix[w][e] = count of edge e in walk w
        # We need: for each walk w, sum_e (count[w][e] * voltage[e]) != 0 mod m
        
        # Try different lift sizes
        for m in [10, 12, 14, 16]:
            if time.time() > deadline - 10:
                break
            
            n_vertices = n_base * m
            
            # Build sparse constraint representation
            # For each walk, store list of (edge_idx, count)
            walk_edge_counts = []
            for w in all_walks:
                counts = {}
                for e in w:
                    counts[e] = counts.get(e, 0) + 1
                walk_edge_counts.append(counts)
            
            # For each edge, list of walk indices that use it
            edge_to_walks = [[] for _ in range(n_edges)]
            for wi, counts in enumerate(walk_edge_counts):
                for e in counts:
                    edge_to_walks[e].append(wi)
            
            # Variable ordering: most constrained first
            var_order = sorted(range(n_edges), key=lambda e: -len(edge_to_walks[e]))
            
            voltages = [-1] * n_edges
            walk_sums = [0] * n_walks  # current sum for each walk
            walk_assigned = [0] * n_walks  # how many edges of each walk are assigned
            walk_total = [len(counts) for counts in walk_edge_counts]
            
            def check_walks_for_edge(e, val):
                """Check if assigning val to edge e violates any constraint"""
                for wi in edge_to_walks[e]:
                    counts = walk_edge_counts[wi]
                    # Check if all edges in this walk are now assigned
                    if walk_assigned[wi] + 1 == walk_total[wi]:
                        new_sum = walk_sums[wi] + val
                        if new_sum % m == 0:
                            return False
                return True
            
            def assign_edge(e, val):
                voltages[e] = val
                for wi in edge_to_walks[e]:
                    walk_sums[wi] += val
                    walk_assigned[wi] += 1
            
            def unassign_edge(e):
                val = voltages[e]
                voltages[e] = -1
                for wi in edge_to_walks[e]:
                    walk_sums[wi] -= val
                    walk_assigned[wi] -= 1
            
            def backtrack(idx):
                if idx == n_edges:
                    return True
                
                if time.time() > deadline - 5:
                    return False
                
                e = var_order[idx]
                
                # Try values in a smart order: prefer values that don't immediately violate
                vals = list(range(m))
                random.shuffle(vals)
                
                for val in vals:
                    if check_walks_for_edge(e, val):
                        assign_edge(e, val)
                        if backtrack(idx + 1):
                            return True
                        unassign_edge(e)
                
                return False
            
            random.seed(42 + m)
            found = False
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                voltages = [-1] * n_edges
                walk_sums = [0] * n_walks
                walk_assigned = [0] * n_walks
                
                if backtrack(0):
                    found = True
                    break
            
            if found:
                # Build lift edges
                lift_edges = []
                for u, v in base_edges:
                    e_idx = edge_index[(u, v)]
                    vu = voltages[e_idx]
                    for i in range(m):
                        lift_edges.append((u*m + i, v*m + (i + vu) % m))
                
                # Verify girth using BFS
                adj_lift = [[] for _ in range(n_vertices)]
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                min_girth = 999
                for start in range(n_vertices):
                    dist = [-1] * n_vertices
                    parent = [-1] * n_vertices
                    dist[start] = 0
                    queue = [start]
                    qi = 0
                    while qi < len(queue):
                        cur = queue[qi]
                        qi += 1
                        for nb in adj_lift[cur]:
                            if dist[nb] == -1:
                                dist[nb] = dist[cur] + 1
                                parent[nb] = cur
                                queue.append(nb)
                            elif parent[nb] != cur and parent[cur] != nb:
                                cycle_len = dist[nb] + dist[cur] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                        if min_girth < g:
                            break
                    if min_girth < g:
                        break
                
                if min_girth >= g:
                    return n_vertices, lift_edges
        
    # Fallback: LUW construction
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
