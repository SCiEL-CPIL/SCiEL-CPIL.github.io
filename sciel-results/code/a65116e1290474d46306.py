import numpy as np
import random
import time
from itertools import combinations

def construct(k, g):
    # Construct PG(2,3) incidence graph
    # Points: 13, Lines: 13
    # Total 26 vertices.
    
    # Generate points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                # Canonicalize: first non-zero is 1
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
    
    # Generate lines
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                # Canonicalize normal
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph
    # Vertices 0..12 are points, 13..25 are lines.
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    m = 10
    n_vertices = 26 * m
    
    # Pre-compute 6-cycles and 8-cycles in base graph
    # A cycle is a list of edges. We need the sum of voltages on these edges.
    # To make it efficient, we represent a cycle by the sequence of vertices.
    # Since the graph is small, we can enumerate all simple cycles of length 6 and 8.
    
    cycles_6 = []
    cycles_8 = []
    
    # Enumerate 6-cycles
    # Fix start vertex 0 to avoid duplicates, but we need all unique cycles.
    # Better: enumerate all paths of length 5 that close to start.
    # To avoid duplicates, we can fix the smallest vertex in the cycle to be the start,
    # and the second smallest to be the second node?
    # For 26 vertices, total 6-cycles is manageable.
    
    def find_cycles(length):
        cycles = []
        # Use DFS
        # To reduce duplicates, we can enforce that the start vertex is the smallest
        # in the cycle, and the next vertex is smaller than the previous? 
        # Or just collect all and deduplicate by frozenset of edges.
        
        for start in range(26):
            stack = [(start, [start])]
            while stack:
                cur, path = stack.pop()
                if len(path) == length:
                    # Check if it closes
                    if start in adj[cur]:
                        # Ensure we don't count duplicates
                        # Canonical form: sort vertices?
                        # Since it's a cycle, the set of vertices is unique.
                        # But different cycles can have same vertex set? No, simple graph.
                        # Wait, a set of 6 vertices might form multiple 6-cycles?
                        # In a 4-regular graph, maybe.
                        # Let's just store the tuple of vertices in order.
                        # And deduplicate by frozenset of edges.
                        edges_in_cycle = set()
                        for i in range(length):
                            u = path[i]
                            v = path[(i+1)%length]
                            edges_in_cycle.add((min(u,v), max(u,v)))
                        if frozenset(edges_in_cycle) not in cycles:
                            cycles.append(frozenset(edges_in_cycle))
                    continue
                
                # Prune: if len(path) > length, stop
                # Optimization: only extend if we haven't visited
                for nb in adj[cur]:
                    if nb not in path:
                        # Pruning: to avoid symmetric duplicates, 
                        # we can require that the start is the smallest vertex in the path so far?
                        # No, that's hard.
                        # Given small size, just run it.
                        # But we need to ensure we don't miss cycles where start isn't smallest.
                        # Actually, if we iterate all starts, we will find all cycles multiple times.
                        # Deduplication by edge set handles this.
                        stack.append((nb, path + [nb]))
        return cycles

    # This DFS might be slow if not pruned.
    # Let's use a simpler enumeration for 6-cycles in bipartite graph.
    # P-L-P-L-P-L-P
    # Fix P1.
    # L1 in adj[P1]
    # P2 in adj[L1], P2 != P1
    # L2 in adj[P2], L2 != L1
    # P3 in adj[L2], P3 != P1, P2
    # L3 in adj[P3], L3 != L2, and L3 must be in adj[P1]
    
    edge_sets_6 = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 == p1 or p3 == p2: continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            if p1 in adj[l3]:
                                # Cycle: p1-l1-p2-l2-p3-l3-p1
                                edges = [
                                    (min(p1,l1), max(p1,l1)),
                                    (min(l1,p2), max(l1,p2)),
                                    (min(p2,l2), max(p2,l2)),
                                    (min(l2,p3), max(l2,p3)),
                                    (min(p3,l3), max(p3,l3)),
                                    (min(l3,p1), max(l3,p1))
                                ]
                                edge_sets_6.add(frozenset(edges))
    
    # 8-cycles: P-L-P-L-P-L-P-L-P
    # This is more complex. 
    # However, if we ensure no 6-cycle has sum 0, and we pick voltages randomly, 
    # the probability of an 8-cycle having sum 0 is low.
    # We can just check 8-cycles in the final verification step if we find a candidate.
    # Or we can enumerate them too.
    
    # Let's try to solve the voltage assignment using random restarts with the 6-cycle constraints.
    # If a 6-cycle constraint is satisfied, the 8-cycle constraint is likely satisfied too.
    
    def check_validity(voltages):
        # Check 6-cycles
        for edges in edge_sets_6:
            s = 0
            for u, v in edges:
                # Voltages are stored as (u, v) -> val
                # We need to sum directed voltages along the cycle.
                # The cycle is undirected in the set. We need to orient it.
                # This is tricky.
                # Let's store the cycle as a path of vertices to orient it.
                pass
        
        # Better approach: Store cycles as vertex sequences.
        return True

    # Let's restart the cycle enumeration to store vertex sequences.
    cycles_6_seq = []
    seen_edge_sets = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 == p1 or p3 == p2: continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            if p1 in adj[l3]:
                                seq = (p1, l1, p2, l2, p3, l3)
                                edges = [
                                    (min(p1,l1), max(p1,l1)),
                                    (min(l1,p2), max(l1,p2)),
                                    (min(p2,l2), max(p2,l2)),
                                    (min(l2,p3), max(l2,p3)),
                                    (min(p3,l3), max(p3,l3)),
                                    (min(l3,p1), max(l3,p1))
                                ]
                                fs = frozenset(edges)
                                if fs not in seen_edge_sets:
                                    seen_edge_sets.add(fs)
                                    cycles_6_seq.append(seq)
    
    # Now search for voltages
    # We assign voltages to edges.
    # Edge (u, v) has voltage v_uv. v_vu = -v_uv.
    # Sum along seq p1->l1->p2->l2->p3->l3->p1 is:
    # v(p1,l1) + v(l1,p2) + v(p2,l2) + v(l2,p3) + v(p3,l3) + v(l3,p1)
    
    valid_assignment = None
    for _ in range(5000):
        voltages = {}
        for u, v in base_edges:
            val = random.randint(0, m-1)
            voltages[(u, v)] = val
            voltages[(v, u)] = (-val) % m
            
        valid = True
        for seq in cycles_6_seq:
            s = 0
            for i in range(len(seq)):
                u = seq[i]
                v = seq[(i+1)%len(seq)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if valid:
            # Check 8-cycles?
            # We can build the lift and check girth via BFS, which is fast for 260 vertices.
            lift_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(m):
                    lift_edges.append((u*m + i, v*m + (i + vu) % m))
            
            # Verify girth
            adj_lift = {i: [] for i in range(n_vertices)}
            for u, v in lift_edges:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                
            min_girth = 999
            for start in range(n_vertices):
                dist = {start: 0}
                parent = {start: None}
                queue = [start]
                while queue:
                    cur = queue.pop(0)
                    for nb in adj_lift[cur]:
                        if nb not in dist:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            queue.append(nb)
                        elif parent[nb] != cur and parent[cur] != nb:
                            cycle_len = dist[nb] + dist[cur] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
            
            if min_girth >= g:
                return n_vertices, lift_edges
                
    # Fallback to LUW
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
