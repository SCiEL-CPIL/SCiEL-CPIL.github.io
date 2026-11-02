import random
import time
from collections import deque

def construct(k, g):
    t_start = time.time()
    DEADLINE = t_start + 145  # Leave 5s safety margin
    
    # Fallback to seed construction if new method fails or is not applicable
    def seed_construct():
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    def verify_girth(adj, n, required_girth):
        half_depth = (required_girth - 1) // 2
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != i and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # New Method: Voltage Lift
    def solve_voltage_lift(base_edges, n_base, M, max_walk_len):
        """
        Finds voltages for base_edges in Z_M such that no closed walk of length < max_walk_len sums to 0.
        Returns list of voltages for edges (index in base_edges) or None.
        """
        n_e = len(base_edges)
        # Map edge index to (u, v)
        # We need to handle directions.
        # Let's assign a voltage to each directed edge?
        # No, just undirected edges with $v_{uv} = -v_{vu}$.
        # We store voltage for edge i as $v_i$.
        # If walk goes u->v and edge is (u,v), add $v_i$. If (v,u), add $-v_i$.
        
        # Precompute walks
        # Walks are sequences of vertices.
        # We need to check lengths 3 to max_walk_len-1.
        
        # Generate all closed walks
        walks = [] # List of (list of edge_indices, list of signs)
        
        # DFS to generate walks
        # Start from each vertex
        # To avoid duplicates, we can just generate all and check?
        # Number of walks is manageable.
        
        def gen_walks(start, current, length, max_len, path):
            if length == 0:
                if current == start:
                    # Store the walk
                    # path is list of vertices [v0, v1, ..., vk]
                    # Edges are (v0,v1), (v1,v2), ...
                    edge_indices = []
                    signs = []
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        # Find edge index
                        # We need a fast lookup
                        # edge_map[(u,v)] = (idx, sign)
                        # where sign is 1 if (u,v) matches base_edges[idx] order, -1 if reversed
                        if (u,v) in edge_map:
                            idx, s = edge_map[(u,v)]
                            edge_indices.append(idx)
                            signs.append(s)
                        else:
                            return # Should not happen
                    walks.append((edge_indices, signs))
                return
            
            for v in range(n_base):
                if v == path[-1]:
                    continue
                # Check if edge exists
                if (path[-1], v) in edge_map or (v, path[-1]) in edge_map:
                    gen_walks(start, v, length-1, max_len, path + [v])

        # Build edge map
        edge_map = {}
        for idx, (u, v) in enumerate(base_edges):
            edge_map[(u, v)] = (idx, 1)
            edge_map[(v, u)] = (idx, -1)
            
        # Generate walks of length 3 to max_walk_len-1
        for start in range(n_base):
            for L in range(3, max_walk_len):
                gen_walks(start, start, L, L, [start])
                
        # Incremental search
        # sums[i] for each walk i
        sums = [0] * len(walks)
        bad = set()
        
        # Initialize voltages
        voltages = [random.randrange(M) for _ in range(n_e)]
        
        # Compute initial sums
        for i, (e_idxs, signs) in enumerate(walks):
            s = 0
            for j in range(len(e_idxs)):
                s += signs[j] * voltages[e_idxs[j]]
            sums[i] = s % M
            if sums[i] == 0:
                bad.add(i)
                
        # Map edge to walks
        # edge_walks[e] = list of (walk_idx, sign)
        edge_walks = [[] for _ in range(n_e)]
        for i, (e_idxs, signs) in enumerate(walks):
            for j in range(len(e_idxs)):
                e = e_idxs[j]
                s = signs[j]
                edge_walks[e].append((i, s))
                
        rng = random.Random(42)
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                return None
            if not bad:
                return voltages
                
            # Pick a random bad walk
            bad_list = list(bad)
            wi = bad_list[rng.randrange(len(bad_list))]
            e_idxs, signs = walks[wi]
            
            # Pick a random edge in this walk to change
            # Prefer edges with high degree in constraints? Just random for now.
            pos = rng.randrange(len(e_idxs))
            e = e_idxs[pos]
            
            # Change voltage
            old_v = voltages[e]
            new_v = rng.randrange(M)
            if new_v == old_v:
                continue
            delta = (new_v - old_v) % M
            voltages[e] = new_v
            
            # Update sums for walks containing e
            for (walk_idx, sign) in edge_walks[e]:
                old_sum = sums[walk_idx]
                new_sum = (old_sum + sign * delta) % M
                sums[walk_idx] = new_sum
                
                if old_sum == 0 and new_sum != 0:
                    bad.discard(walk_idx)
                elif old_sum != 0 and new_sum == 0:
                    bad.add(walk_idx)
                    
        return None

    # Base Graph K5
    n_base = 5
    base_edges_k5 = [(i, j) for i in range(5) for j in range(i+1, 5)]
    
    # Try K5 with M starting from 49
    found = False
    for M in range(49, 60):
        if time.time() > DEADLINE - 10:
            break
        voltages = solve_voltage_lift(base_edges_k5, n_base, M, g)
        if voltages is not None:
            # Build the graph
            n = n_base * M
            edges = []
            for u in range(n_base):
                for v in range(n_base):
                    if u == v:
                        continue
                    # Find edge index
                    # We need the voltage for (u,v)
                    # base_edges_k5 has (i,j) with i<j
                    if u < v:
                        idx = base_edges_k5.index((u, v))
                        v_uv = voltages[idx]
                    else:
                        idx = base_edges_k5.index((v, u))
                        v_uv = -voltages[idx]
                        
                    for g_val in range(M):
                        u_lift = u * M + g_val
                        v_lift = v * M + ((g_val + v_uv) % M)
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
            found = True
            break
            
    if not found:
        # Try B6 (K6 minus perfect matching)
        n_base = 6
        # Perfect matching: (0,1), (2,3), (4,5)
        pm = {(0,1), (1,0), (2,3), (3,2), (4,5), (5,4)}
        base_edges_b6 = [(i, j) for i in range(6) for j in range(i+1, 6) if (i,j) not in pm]
        
        for M in range(40, 50):
            if time.time() > DEADLINE - 5:
                break
            voltages = solve_voltage_lift(base_edges_b6, n_base, M, g)
            if voltages is not None:
                n = n_base * M
                edges = []
                for u in range(n_base):
                    for v in range(n_base):
                        if u == v:
                            continue
                        if (u, v) in pm:
                            continue # No edge in base
                            
                        if u < v:
                            idx = base_edges_b6.index((u, v))
                            v_uv = voltages[idx]
                        else:
                            idx = base_edges_b6.index((v, u))
                            v_uv = -voltages[idx]
                            
                        for g_val in range(M):
                            u_lift = u * M + g_val
                            v_lift = v * M + ((g_val + v_uv) % M)
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                found = True
                break
                
    if found:
        adj = build_adj(n, edges)
        if verify_girth(adj, n, g):
            return n, edges
        else:
            # Fallback
            n, edges = seed_construct()
            return n, edges
    else:
        n, edges = seed_construct()
        return n, edges
