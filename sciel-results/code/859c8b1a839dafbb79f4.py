import time
import random
import itertools
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def check_girth(n, edges, g, time_limit=None):
        """Check if graph has girth >= g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    def build_luw_graph(q, d, rule, time_limit=None):
        """Build LUW graph with given parameters."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

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
            if time_limit and time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def generate_rules(d, variant=0):
        base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                if c % 2 == 0:
                    base_rule[c] = (1, c - 2)
                else:
                    base_rule[c] = (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if c % 3 == 0:
                    base_rule[c] = (1, c - 3)
                elif c % 3 == 1:
                    base_rule[c] = (c - 1, 1)
                else:
                    base_rule[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                offset = 1 + (c % 2)
                base_rule[c] = (1, c - offset)
        elif variant == 4:
            for c in range(5, d + 1):
                if c % 4 == 0:
                    base_rule[c] = (1, c - 1)
                elif c % 4 == 1:
                    base_rule[c] = (c - 1, 1)
                elif c % 4 == 2:
                    base_rule[c] = (1, c - 2)
                else:
                    base_rule[c] = (c - 2, 1)
        else:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base_rule

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best_result = None
        best_n = float('inf')
        for variant in range(5):
            if time.time() > deadline - 15:
                break
            rule = generate_rules(d, variant)
            result = build_luw_graph(q, d, rule, time_limit=deadline - 15)
            if result is None:
                continue
            n, edges = result
            if n >= best_n:
                continue
            girth_check = check_girth(n, edges, g, time_limit=deadline - 10)
            if girth_check == -1:
                continue
            if girth_check >= g:
                best_result = (n, edges)
                best_n = n
        if best_result is None:
            for d_try in range(d + 1, d + 10):
                if time.time() > deadline - 10:
                    break
                for variant in range(3):
                    rule = generate_rules(d_try, variant)
                    result = build_luw_graph(q, d_try, rule, time_limit=deadline - 10)
                    if result is None:
                        continue
                    n, edges = result
                    girth_check = check_girth(n, edges, g, time_limit=deadline - 5)
                    if girth_check == -1:
                        continue
                    if girth_check >= g:
                        best_result = (n, edges)
                        break
                if best_result is not None:
                    break
        if best_result is None:
            rule = generate_rules(d, 0)
            result = build_luw_graph(q, d, rule)
            if result is not None:
                best_result = result
        return best_result

    def voltage_lift_k33(m, voltages, time_limit=None):
        """
        Build a 3-regular graph by voltage lift of K_{3,3} over Z_m.
        
        K_{3,3} has vertices A0,A1,A2 (left) and B0,B1,B2 (right).
        Edges: Ai-Bj for all i,j in {0,1,2}.
        
        Voltage assignment: each edge (Ai, Bj) gets a voltage v_{i,j} in Z_m.
        The lift has 6m vertices: (side, base, coset) where side in {0,1},
        base in {0,1,2}, coset in Z_m.
        
        Vertex (0, i, a) connects to (1, j, a + v_{i,j} mod m) for all j.
        Vertex (1, j, b) connects to (0, i, b - v_{i,j} mod m) for all i.
        """
        n = 6 * m
        edges = []
        seen = set()
        
        for i in range(3):
            for j in range(3):
                v = voltages[i * 3 + j]
                for a in range(m):
                    # Edge from (0,i,a) to (1,j, a+v mod m)
                    u = i * m + a
                    w = 3 * m + j * m + ((a + v) % m)
                    if u < w:
                        edges.append((u, w))
                    else:
                        edges.append((w, u))
        
        return n, edges

    def girth_of_lift(m, voltages, g_target, time_limit=None):
        """
        Compute the girth of the voltage lift of K_{3,3} over Z_m.
        Uses BFS in the non-backtracking sense to find shortest cycle.
        
        For a voltage graph, cycles correspond to closed walks in the base
        whose voltage sum is 0 mod m. We can compute girth by BFS in the
        lift graph directly, but that's expensive for large m.
        
        Alternative: BFS from each vertex in the lift, tracking distance.
        For 6m vertices with degree 3, BFS from each vertex is O(6m * 6m) = O(36m^2).
        For m up to ~200, that's ~1.4M operations per BFS, 6m BFS calls = ~864M total.
        That's too slow. Let's use a smarter approach.
        
        Actually, let's just do BFS in the lift but with early termination.
        """
        n = 6 * m
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = voltages[i * 3 + j]
                for a in range(m):
                    u = i * m + a
                    w = 3 * m + j * m + ((a + v) % m)
                    adj[u].append(w)
                    adj[w].append(u)
        
        min_cycle = float('inf')
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g_target:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g_target:
                return min_cycle
        return min_cycle

    def fast_girth_check(m, voltages, g_target, time_limit=None):
        """
        Faster girth check for K_{3,3} voltage lift.
        """
        if m <= 120:
            return girth_of_lift(m, voltages, g_target, time_limit)
        if m <= 200:
            return girth_of_lift(m, voltages, g_target, time_limit)
        return -1

    def _petersen_voltage_lift_optimize(k, g, deadline):
        """
        Build a 3-regular graph via voltage lift of the Petersen graph over Z_m.
        The Petersen graph has 10 vertices, 15 edges, is 3-regular with girth 5.
        A voltage lift over Z_m gives 10m vertices, 15m edges, 3-regular.
        We search for voltage assignments that give girth >= g.
        """
        # Petersen graph: vertices 0-9
        # Outer cycle: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        petersen_edges = [
            (0,1), (1,2), (2,3), (3,4), (4,0),  # outer 5-cycle
            (5,7), (7,9), (9,6), (6,8), (8,5),  # inner 5-pointed star
            (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
        ]
        num_edges = len(petersen_edges)
        
        best_result = None
        best_n = float('inf')
        
        # Try different values of m
        # n = 10*m vertices. We want girth >= g.
        # Moore bound for n(3,16) is 512, so m >= 52 gives n >= 520.
        # Start from reasonable m values.
        
        for m in range(60, 200):
            if time.time() > deadline - 25:
                break
            
            n = 10 * m
            if n >= best_n:
                break
            
            # Try random voltage assignments
            found = False
            max_attempts = 200
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 25:
                    break
                
                voltages = [random.randint(0, m-1) for _ in range(num_edges)]
                
                # Quick check: build adjacency and check girth
                # For Petersen lift, we can do BFS efficiently
                adj = [[] for _ in range(n)]
                for edge_idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[edge_idx]
                    for a in range(m):
                        # Vertex u in coset a connects to v in coset a+vol
                        uu = u * m + a
                        vv = v * m + ((a + vol) % m)
                        adj[uu].append(vv)
                        adj[vv].append(uu)
                
                # Check girth via BFS from a few vertices (not all, for speed)
                # If we find a short cycle, reject; if we don't find one in limited BFS, accept tentatively
                girth_ok = True
                # Check from first few vertices
                check_vertices = list(range(0, n, max(1, n // 10)))
                for s in check_vertices:
                    if time.time() > deadline - 25:
                        break
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[s] = 0
                    queue = [s]
                    qi = 0
                    while qi < len(queue):
                        node = queue[qi]
                        qi += 1
                        for nb in adj[node]:
                            if nb == parent[node]:
                                continue
                            if dist[nb] != -1:
                                cycle_len = dist[node] + dist[nb] + 1
                                if cycle_len < g:
                                    girth_ok = False
                                    break
                            else:
                                dist[nb] = dist[node] + 1
                                parent[nb] = node
                                queue.append(nb)
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # Full verification
                    full_girth = check_girth(n, [(min(u*m+a, v*m+((a+vol)%m)), max(u*m+a, v*m+((a+vol)%m))) 
                                                 for edge_idx, (u, v) in enumerate(petersen_edges) 
                                                 for a in range(m) 
                                                 for vol in [voltages[edge_idx]]], g, time_limit=deadline - 10)
                    if full_girth != -1 and full_girth >= g:
                        # Build edge list properly
                        edges = []
                        seen_edges = set()
                        for edge_idx, (u, v) in enumerate(petersen_edges):
                            vol = voltages[edge_idx]
                            for a in range(m):
                                uu = u * m + a
                                vv = v * m + ((a + vol) % m)
                                edge = (min(uu, vv), max(uu, vv))
                                if edge not in seen_edges:
                                    seen_edges.add(edge)
                                    edges.append(edge)
                        best_result = (n, edges)
                        best_n = n
                        found = True
                        break
            
            if found:
                # Found valid graph, try smaller m
                # Since we're going in increasing order, try to see if smaller m works
                # Actually we should try smaller m first. Let's restructure.
                break
        
        # If we didn't find with m in [60,200), try smaller m range
        if best_result is None:
            for m in range(52, 60):
                if time.time() > deadline - 25:
                    break
                n = 10 * m
                max_attempts = 100
                for attempt in range(max_attempts):
                    if time.time() > deadline - 25:
                        break
                    voltages = [random.randint(0, m-1) for _ in range(num_edges)]
                    adj = [[] for _ in range(n)]
                    for edge_idx, (u, v) in enumerate(petersen_edges):
                        vol = voltages[edge_idx]
                        for a in range(m):
                            uu = u * m + a
                            vv = v * m + ((a + vol) % m)
                            adj[uu].append(vv)
                            adj[vv].append(uu)
                    
                    girth_ok = True
                    check_vertices = list(range(0, n, max(1, n // 10)))
                    for s in check_vertices:
                        if time.time() > deadline - 25:
                            break
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[s] = 0
                        queue = [s]
                        qi = 0
                        while qi < len(queue):
                            node = queue[qi]
                            qi += 1
                            for nb in adj[node]:
                                if nb == parent[node]:
                                    continue
                                if dist[nb] != -1:
                                    cycle_len = dist[node] + dist[nb] + 1
                                    if cycle_len < g:
                                        girth_ok = False
                                        break
                                else:
                                    dist[nb] = dist[node] + 1
                                    parent[nb] = node
                                    queue.append(nb)
                            if not girth_ok:
                                break
                        if not girth_ok:
                            break
                    
                    if girth_ok:
                        full_girth = check_girth(n, [(min(u*m+a, v*m+((a+voltages[edge_idx])%m)), max(u*m+a, v*m+((a+voltages[edge_idx])%m))) 
                                                     for edge_idx, (u, v) in enumerate(petersen_edges) 
                                                     for a in range(m)], g, time_limit=deadline - 10)
                        if full_girth != -1 and full_girth >= g:
                            edges = []
                            seen_edges = set()
                            for edge_idx, (u, v) in enumerate(petersen_edges):
                                vol = voltages[edge_idx]
                                for a in range(m):
                                    uu = u * m + a
                                    vv = v * m + ((a + vol) % m)
                                    edge = (min(uu, vv), max(uu, vv))
                                    if edge not in seen_edges:
                                        seen_edges.add(edge)
                                        edges.append(edge)
                            best_result = (n, edges)
                            break
                if best_result is not None:
                    break
        
        return best_result

    if k != 3:
        return luw_construct(k, g)
    
    # For k=3, try Petersen voltage lift first (potentially much smaller)
    petersen_result = _petersen_voltage_lift_optimize(k, g, deadline)
    
    # Get LUW baseline as fallback
    luw_result = luw_construct(k, g)
    
    # Choose the better result
    if petersen_result is not None and luw_result is not None:
        if petersen_result[0] < luw_result[0]:
            return petersen_result
        else:
            return luw_result
    elif petersen_result is not None:
        return petersen_result
    elif luw_result is not None:
        return luw_result
    else:
        return 0, []
