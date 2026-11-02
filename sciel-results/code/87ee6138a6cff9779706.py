import time
import sys
import math
import random
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Moore bound for (k, g)
    if g % 2 == 0:
        moore_bound = 2 * k * ((k - 1) ** (g // 2 - 1) - 1) // (k - 2)
    else:
        if k > 2:
            moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2 - 1) * (k - 1) + 1) // (k - 2)
        else:
            moore_bound = g
    
    # Strategy 1: Circulant Graph Search (from current program)
    # Precompute non-backtracking walk coefficients
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

    # Strategy 2: Random Lift of K_{4,4} with M=40 (n=320)
    # This is a bipartite graph, so girth is even. We check 4, 6, 8.
    if k == 4 and g == 10:
        if time.time() < deadline - 10:
            M = 40
            # Base graph K_{4,4}
            # Vertices 0-3 (U), 4-7 (V)
            # Edges (u, v+4)
            # Voltages w[u][v] for u in 0..3, v in 0..3
            # Lifted graph size 5*? No, 8*M = 320.
            
            def check_girth_lift(voltages, M, g):
                # voltages is 4x4 matrix
                # Graph vertices: (u, i) where u in 0..7, i in 0..M-1
                # Adjacency:
                # If u < 4 (U-side), neighbors are (v+4, (i + voltages[u][v]) % M) for v in 0..3
                # If u >= 4 (V-side), let v = u-4. Neighbors are (u', (i - voltages[u'][v]) % M) for u' in 0..3
                # Note: Edge direction matters for voltage consistency.
                # Let's define voltage w_uv as the voltage from u to v.
                # In K_{4,4}, edge is undirected. We assign w[u][v].
                # From U u to V v: i' = (i + w[u][v]) % M
                # From V v to U u: i' = (i - w[u][v]) % M
                
                # Check for cycles of length L < g
                # Since it's bipartite, L is even.
                # We do BFS from each vertex up to depth g-1.
                # n = 320. Depth 9.
                # To speed up, we can just check if any vertex is revisited in a non-backtracking walk.
                
                n_vertices = 8 * M
                # Precompute neighbors for speed? 
                # Or just generate on the fly.
                
                # Optimization: Check from a subset of vertices? No, must be all.
                # But we can stop early if found.
                
                for start_v in range(n_vertices):
                    # BFS
                    # State: (vertex, last_vertex)
                    # We need to avoid backtracking.
                    # visited set of (vertex, last_vertex) is too big?
                    # 320 * 320 = 100k states. Manageable.
                    # But we only go up to depth 9.
                    # Number of non-backtracking walks of length 9 is 4*3^8 ~ 25k.
                    # Total walks from all vertices: 320 * 25k = 8M.
                    # This is fine.
                    
                    visited = set()
                    # Queue: (current_v, last_v, depth)
                    # We start by taking one step.
                    u = start_v // M
                    i = start_v % M
                    
                    # Determine neighbors of start_v
                    if u < 4:
                        # U-side
                        for v in range(4):
                            nv = (v + 4) * M + (i + voltages[u][v]) % M
                            visited.add((nv, start_v))
                    else:
                        # V-side
                        v = u - 4
                        for u_prime in range(4):
                            nv = u_prime * M + (i - voltages[u_prime][v]) % M
                            visited.add((nv, start_v))
                    
                    # Expand
                    # We need to check if we close a cycle.
                    # A cycle closes if we return to start_v with last_v != start_v (non-backtracking).
                    # Actually, if we return to start_v, and the last vertex is not the one we came from (which is not defined for start), 
                    # wait.
                    # Standard BFS for girth:
                    # Keep track of (vertex, parent) to avoid immediate backtracking.
                    # If we reach (start_v, parent) where parent != start_v, we found a cycle of length depth+1.
                    
                    queue = []
                    for nv, _ in list(visited):
                        queue.append((nv, start_v, 1))
                        
                    while queue:
                        if time.time() > deadline - 5:
                            return False # Timeout, assume invalid to be safe? Or break outer?
                        curr, parent, depth = queue.pop(0)
                        
                        if curr == start_v and depth > 1:
                            # Cycle found of length depth
                            if depth < g:
                                return False
                            # If depth >= g, we don't care for this start, but we must ensure NO cycle < g.
                            # So if depth < g, return False.
                            
                        if depth + 1 >= g:
                            continue
                            
                        u_c = curr // M
                        i_c = curr % M
                        
                        if u_c < 4:
                            for v in range(4):
                                nv = (v + 4) * M + (i_c + voltages[u_c][v]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) in visited:
                                    continue
                                visited.add((nv, curr))
                                queue.append((nv, curr, depth + 1))
                        else:
                            v_c = u_c - 4
                            for u_prime in range(4):
                                nv = u_prime * M + (i_c - voltages[u_prime][v_c]) % M
                                if nv == parent:
                                    continue
                                if (nv, curr) in visited:
                                    continue
                                visited.add((nv, curr))
                                queue.append((nv, curr, depth + 1))
                                
                    # If loop finishes, no short cycle from start_v
                return True

            # Random search for voltages
            for _ in range(500): # 500 trials
                if time.time() > deadline - 10:
                    break
                # Random voltages
                volts = [[random.randint(0, M-1) for _ in range(4)] for _ in range(4)]
                
                if check_girth_lift(volts, M, g):
                    # Construct edges
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
                    
            # Local Search if random failed? 
            # Given time, maybe just try more randoms or different M?
            # If 320 fails, try 325 (M=40.625? No, M must be integer. 325 is not 8*M. 325=5*65. Base K5, M=65).
            
            # Try K5 Lift with M=64 (n=320) as well?
            # K5 is not bipartite. Girth check is harder (odd cycles).
            # Let's stick to K4,4 for now.
            
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
