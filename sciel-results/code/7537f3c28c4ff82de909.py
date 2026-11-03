import time
import sys
import math
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound for (k, g)
    if g % 2 == 0:
        moore_bound = 2 * k * ((k - 1) ** (g // 2 - 1) - 1) // (k - 2)
    else:
        moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2 - 1) * (k - 1) + 1) // (k - 2) if k > 2 else g
    
    # Strategy 1: Circulant Graph Search
    bad_coeffs = set()
    if g > 3:
        states = {
            (1, 1, 0),
            (-1, -1, 0),
            (2, 0, 1),
            (-2, 0, -1)
        }
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    bad_coeffs.add((c1, c2))
    
    # Search for n starting from Moore bound
    max_n_search = moore_bound + 20
    for n in range(moore_bound, max_n_search + 1):
        if time.time() > deadline - 20:
            break
        for a in range(1, n // 2):
            if time.time() > deadline - 15:
                break
            for b in range(a + 1, n // 2 + 1):
                if math.gcd(math.gcd(a, b), n) != 1:
                    continue
                valid = True
                for c1, c2 in bad_coeffs:
                    if (c1 * a + c2 * b) % n == 0:
                        valid = False
                        break
                if valid:
                    unique_edges = set()
                    for i in range(n):
                        j1 = (i + a) % n
                        j2 = (i + b) % n
                        if i < j1: unique_edges.add((i, j1))
                        else: unique_edges.add((j1, i))
                        if i < j2: unique_edges.add((i, j2))
                        else: unique_edges.add((j2, i))
                    return n, list(unique_edges)

    # Strategy 2: Voltage Lift with Optimized Search
    if k == 4 and g == 10:
        if time.time() < deadline - 5:
            # Use Desargues graph (bipartite, 28 vertices) as base
            # Desargues graph: vertices 0-13 (U), 14-27 (V)
            # Construct from projective plane of order 3
            
            # Build Desargues graph adjacency
            desargues_adj = [[] for _ in range(28)]
            
            # Projective plane of order 3: 13 points, 13 lines, each line has 4 points
            # Points: 0-12, Lines: 0-12
            # Each point is on 4 lines, each line has 4 points
            
            # Standard representation
            points_on_lines = [
                [0, 1, 2, 3],      # Line 0
                [0, 4, 5, 6],      # Line 1
                [0, 7, 8, 9],      # Line 2
                [0, 10, 11, 12],   # Line 3
                [1, 4, 7, 10],     # Line 4
                [1, 5, 8, 11],     # Line 5
                [1, 6, 9, 12],     # Line 6
                [2, 4, 8, 12],     # Line 7
                [2, 5, 9, 10],     # Line 8
                [2, 6, 7, 11],     # Line 9
                [3, 4, 9, 11],     # Line 10
                [3, 5, 7, 12],     # Line 11
                [3, 6, 8, 10],     # Line 12
            ]
            
            # Incidence: point i is on line j if i in points_on_lines[j]
            point_lines = [[] for _ in range(13)]
            for j, pts in enumerate(points_on_lines):
                for i in pts:
                    point_lines[i].append(j)
            
            # Desargues graph: U vertices = points (0-12), V vertices = lines (13-25)
            # Wait, that's only 26 vertices. Let me use the standard Desargues graph.
            
            # Actually, let's use a simpler approach: K_{4,4} with optimized voltages
            M = 40  # n = 8*M = 320
            
            def check_girth_fast(voltages, M, g):
                """Fast girth check using BFS from each vertex"""
                n_vertices = 8 * M
                max_depth = g - 1  # We only care about cycles < g
                
                for start_v in range(n_vertices):
                    u = start_v // M
                    i = start_v % M
                    
                    # BFS state: (current_vertex, parent_vertex)
                    visited = set()
                    queue = []
                    
                    # Initialize: neighbors of start_v
                    if u < 4:
                        for v in range(4):
                            nv = (v + 4) * M + (i + voltages[u][v]) % M
                            visited.add((nv, start_v))
                            queue.append((nv, start_v, 1))
                    else:
                        v = u - 4
                        for u_prime in range(4):
                            nv = u_prime * M + (i - voltages[u_prime][v]) % M
                            visited.add((nv, start_v))
                            queue.append((nv, start_v, 1))
                    
                    while queue:
                        curr, parent, depth = queue.pop(0)
                        
                        if curr == start_v and depth > 1:
                            if depth < g:
                                return False
                        
                        if depth + 1 >= g:
                            continue
                        
                        u_c = curr // M
                        i_c = curr % M
                        
                        if u_c < 4:
                            for v in range(4):
                                nv = (v + 4) * M + (i_c + voltages[u_c][v]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) not in visited:
                                    visited.add((nv, curr))
                                    queue.append((nv, curr, depth + 1))
                        else:
                            v_c = u_c - 4
                            for u_prime in range(4):
                                nv = u_prime * M + (i_c - voltages[u_prime][v_c]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) not in visited:
                                    visited.add((nv, curr))
                                    queue.append((nv, curr, depth + 1))
                
                return True
            
            # Simulated annealing for voltage optimization
            def count_short_cycles(voltages, M, g):
                """Count cycles of length < g (for annealing objective)"""
                n_vertices = 8 * M
                max_depth = min(g - 1, 8)  # Only check up to depth 8 for speed
                cycle_count = 0
                
                for start_v in range(n_vertices):
                    u = start_v // M
                    i = start_v % M
                    
                    visited = set()
                    queue = []
                    
                    if u < 4:
                        for v in range(4):
                            nv = (v + 4) * M + (i + voltages[u][v]) % M
                            visited.add((nv, start_v))
                            queue.append((nv, start_v, 1))
                    else:
                        v = u - 4
                        for u_prime in range(4):
                            nv = u_prime * M + (i - voltages[u_prime][v]) % M
                            visited.add((nv, start_v))
                            queue.append((nv, start_v, 1))
                    
                    while queue:
                        curr, parent, depth = queue.pop(0)
                        
                        if curr == start_v and depth > 1:
                            if depth < g:
                                cycle_count += 1
                                return cycle_count  # Early exit for speed
                        
                        if depth + 1 > max_depth:
                            continue
                        
                        u_c = curr // M
                        i_c = curr % M
                        
                        if u_c < 4:
                            for v in range(4):
                                nv = (v + 4) * M + (i_c + voltages[u_c][v]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) not in visited:
                                    visited.add((nv, curr))
                                    queue.append((nv, curr, depth + 1))
                        else:
                            v_c = u_c - 4
                            for u_prime in range(4):
                                nv = u_prime * M + (i_c - voltages[u_prime][v_c]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) not in visited:
                                    visited.add((nv, curr))
                                    queue.append((nv, curr, depth + 1))
                
                return cycle_count
            
            # Try random initialization first
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Random voltages
                volts = [[random.randint(0, M-1) for _ in range(4)] for _ in range(4)]
                
                if check_girth_fast(volts, M, g):
                    edges = set()
                    for u in range(4):
                        for i in range(M):
                            v_start = u * M + i
                            for v in range(4):
                                v_end = (v + 4) * M + (i + volts[u][v]) % M
                                if v_start < v_end:
                                    edges.add((v_start, v_end))
                                else:
                                    edges.add((v_end, v_start))
                    return 320, list(edges)
            
            # If random failed, try local search
            if time.time() < deadline - 10:
                # Start from random
                best_volts = [[random.randint(0, M-1) for _ in range(4)] for _ in range(4)]
                best_score = count_short_cycles(best_volts, M, g)
                
                # Local search: try changing one voltage at a time
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for u in range(4):
                        for v in range(4):
                            old_val = best_volts[u][v]
                            for new_val in range(M):
                                if new_val == old_val:
                                    continue
                                test_volts = [row[:] for row in best_volts]
                                test_volts[u][v] = new_val
                                score = count_short_cycles(test_volts, M, g)
                                if score < best_score:
                                    best_volts = test_volts
                                    best_score = score
                                    improved = True
                                    break
                                elif score == 0:
                                    # Found a solution!
                                    if check_girth_fast(test_volts, M, g):
                                        edges = set()
                                        for uu in range(4):
                                            for i in range(M):
                                                v_start = uu * M + i
                                                for vv in range(4):
                                                    v_end = (vv + 4) * M + (i + test_volts[uu][vv]) % M
                                                    if v_start < v_end:
                                                        edges.add((v_start, v_end))
                                                    else:
                                                        edges.add((v_end, v_start))
                                        return 320, list(edges)
        
    # Fallback: LUW Construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
