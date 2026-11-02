import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def verify_girth(n, edges, target_girth):
        """
        Verify that the graph has no cycle shorter than target_girth.
        Uses BFS from each vertex to find shortest cycle.
        """
        if n == 0:
            return True
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for loops
        for u, v in edges:
            if u == v:
                return False
        
        # Check for repeated edges
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            
        # Check girth using BFS from each vertex
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            min_cycle = float('inf')
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] * 2 >= min_cycle:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
            if min_cycle < target_girth:
                return False
                
        return True

    # Search for a Voltage Graph on K_4,4 over Z_40
    if k == 4 and g == 10:
        n_base = 8
        group_n = 40
        n_total = n_base * group_n
        
        # Base graph K_4,4
        # Vertices 0-3 are U, 4-7 are V
        # Edges (i, 4+j)
        
        def check_voltage_matrix(A):
            """
            A is a 4x4 matrix in Z_40.
            Returns True if the lift has girth >= 10.
            """
            # Build adjacency for the lifted graph implicitly
            # Vertex index: u * 40 + g for u in 0..3
            #               (4+j) * 40 + g for j in 0..3
            
            # Pre-compute neighbors
            # adj[u, g] = [(4+j, g + A[u][j]) for j in 0..3]
            # adj[4+j, g] = [(u, g - A[u][j]) for u in 0..3]
            
            # Check 4-cycles
            # u_i - v_j - u_k - v_l - u_i
            # Sum: A[i][j] - A[k][j] + A[k][l] - A[i][l] != 0
            for i in range(4):
                for k in range(i+1, 4):
                    for j in range(4):
                        for l in range(j+1, 4):
                            s = A[i][j] - A[k][j] + A[k][l] - A[i][l]
                            if s % group_n == 0:
                                return False
            
            # Check 6-cycles
            # u_i - v_j - u_k - v_l - u_m - v_n - u_i
            # i,k,m distinct; j,l,n distinct
            # Sum: A[i][j] - A[k][j] + A[k][l] - A[m][l] + A[m][n] - A[i][n] != 0
            for i in range(4):
                for k in range(4):
                    if i == k: continue
                    for m in range(4):
                        if i == m or k == m: continue
                        for j in range(4):
                            for l in range(4):
                                if j == l: continue
                                for n in range(4):
                                    if j == n or l == n: continue
                                    s = A[i][j] - A[k][j] + A[k][l] - A[m][l] + A[m][n] - A[i][n]
                                    if s % group_n == 0:
                                        return False
                                    
            # Check 8-cycles using BFS on the lifted graph
            # If we pass 4 and 6, we must check 8.
            # Build adjacency list
            adj = [[] for _ in range(n_total)]
            for i in range(4):
                for j in range(4):
                    for g in range(group_n):
                        u_idx = i * group_n + g
                        v_idx = (4+j) * group_n + (g + A[i][j]) % group_n
                        adj[u_idx].append(v_idx)
                        adj[v_idx].append(u_idx)
            
            # BFS from each vertex to find min cycle
            for s in range(n_total):
                dist = [-1] * n_total
                parent = [-1] * n_total
                dist[s] = 0
                queue = [s]
                head = 0
                min_cycle = 10 # We only care if < 10
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    if dist[u] * 2 >= min_cycle:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if parent[u] != v and parent[v] != u:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_cycle:
                                    min_cycle = cycle_len
                if min_cycle < 10:
                    return False
            return True

        # Search for A
        # Fix first row to 0, 1, 2, 3
        A = [[0, 1, 2, 3]]
        
        # Iterate over rows 1, 2, 3
        # To speed up, we can iterate over row 1, then row 2, then row 3
        # But 40^3 = 64000. For each, we check 4-cycles (fast), 6-cycles (slower), 8-cycles (slowest).
        # We can prune early.
        
        count = 0
        for r1 in range(40):
            if time.time() > deadline - 20: break
            for r2 in range(40):
                if time.time() > deadline - 20: break
                for r3 in range(40):
                    if time.time() > deadline - 20: break
                    count += 1
                    A[1] = [0, r1, (r1+1)%40, (r1+2)%40] # Example pattern? No, just random/iterate
                    # Actually, let's just iterate all values for simplicity in the loop structure
                    # But 40^3 is small.
                    
                    # Let's use a structured search:
                    # A[i][j] = i * j * C + D_i + E_j?
                    # No, let's just iterate r1, r2, r3 as full rows?
                    # That's 40^12. Too big.
                    
                    # Let's assume a form: A[i][j] = (i * P + j * Q + R) % 40?
                    # No, that's additive, which fails 4-cycles.
                    
                    # Let's try A[i][j] = (i * j * C) % 40 + D_i?
                    # We already tried i*j.
                    
                    # Let's just pick random matrices?
                    # No, deterministic is better.
                    
                    # Let's iterate over 3 parameters for a structured family.
                    # A[i][j] = (a * i * j + b * i + c * j) % 40?
                    # This is additive in i and j separately?
                    # A[i][j] - A[k][j] = (a(i-k)j + b(i-k)) = (i-k)(aj+b).
                    # A[i][l] - A[k][l] = (i-k)(al+b).
                    # Sum 4-cycle: (i-k)(aj+b) - (i-k)(al+b) = (i-k) a (j-l).
                    # This is 0 if a=0 or j=l or i=k.
                    # So this family fails 4-cycles unless a != 0.
                    # If a != 0, sum is non-zero for distinct i,j,k,l.
                    # So this family passes 4-cycles!
                    
                    # Let's search a, b, c in Z_40.
                    # A[i][j] = (a * i * j + b * i + c * j) % 40
                    # Check 6-cycles and 8-cycles.
                    
                    for a in range(1, 40):
                        if time.time() > deadline - 20: break
                        for b in range(40):
                            if time.time() > deadline - 20: break
                            for c in range(40):
                                if time.time() > deadline - 20: break
                                A = [[(a * i * j + b * i + c * j) % 40 for j in range(4)] for i in range(4)]
                                if check_voltage_matrix(A):
                                    # Construct edges
                                    edges = []
                                    for i in range(4):
                                        for j in range(4):
                                            for g in range(40):
                                                u_idx = i * 40 + g
                                                v_idx = (4+j) * 40 + (g + A[i][j]) % 40
                                                if u_idx < v_idx:
                                                    edges.append((u_idx, v_idx))
                                                else:
                                                    edges.append((v_idx, u_idx))
                                    # Verify
                                    if verify_girth(n_total, edges, g):
                                        return n_total, edges

    # Fallback: LUW Construction with d=4 (2048 vertices)
    if time.time() < deadline - 10:
        q = k
        d = 4
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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        
        n_luw = len(order)
        if verify_girth(n_luw, edges_luw, g):
            return n_luw, edges_luw

    # Final Fallback: Seed LUW with d=5 (8192 vertices)
    q = k
    d = 5
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    
    return len(order), edges_luw
