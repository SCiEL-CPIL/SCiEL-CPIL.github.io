"""Improved program: construct k-regular graph of girth g using optimized Cayley graphs and L-U-W construction."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque
from math import gcd

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Try to find a valid set of generators for a Cayley graph on (S_3)^4.
       This yields n=1296.
    2. If that fails, try L-U-W construction CD(4, 5).
       This yields n=1250. Verify girth.
    3. If that fails, try L-U-W construction CD(5, 5).
       This yields n=6250.
    """
    time_limit = 110  # seconds
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    def verify_girth(n, edges, g):
        """Verify that the graph has girth at least g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for self-loops and duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
        
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        
        return True

    def try_cayley_s3_4(k, g, time_budget):
        """Find 5 generators in (S_3)^4 forming a clique in the compatibility graph."""
        if k != 5 or g != 9:
            return None
            
        start_time = time.time()
        
        # Group: (S_3)^4
        perms = [
            (0,1,2), # 0: id
            (1,0,2), # 1: (12)
            (2,1,0), # 2: (13)
            (0,2,1), # 3: (23)
            (1,2,0), # 4: (123)
            (2,0,1)  # 5: (132)
        ]
        
        def mul_s3(a, b):
            p_a = perms[a]
            p_b = perms[b]
            result = tuple(p_a[p_b[i]] for i in range(3))
            for idx, p in enumerate(perms):
                if p == result:
                    return idx
            return 0
            
        mul_table = [[0]*6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                mul_table[i][j] = mul_s3(i, j)
                
        invol_s3 = [1, 2, 3]
        
        invol_g = []
        for a in invol_s3:
            for b in invol_s3:
                for c in invol_s3:
                    for d in invol_s3:
                        invol_g.append((a,b,c,d))
                        
        def mul_g(u, v):
            return (
                mul_table[u[0]][v[0]],
                mul_table[u[1]][v[1]],
                mul_table[u[2]][v[2]],
                mul_table[u[3]][v[3]]
            )
            
        def order_g(u):
            orders = [1, 2, 2, 2, 3, 3]
            o = 1
            for i in range(4):
                oi = orders[u[i]]
                o = o * oi // gcd(o, oi)
            return o
            
        n_inv = len(invol_g)
        comp = [[False]*n_inv for _ in range(n_inv)]
        
        # Build compatibility graph
        for i in range(n_inv):
            for j in range(i+1, n_inv):
                u = invol_g[i]
                v = invol_g[j]
                w = mul_g(u, v)
                if order_g(w) == 6:
                    comp[i][j] = True
                    comp[j][i] = True
                    
        # Find clique of size 5
        def find_clique(remaining, current, size):
            if size == 5:
                return current
            if not remaining:
                return None
            
            # Simple backtracking
            for idx, node in enumerate(remaining):
                new_current = current + [node]
                new_remaining = [n for n in remaining[idx+1:] if comp[node][n]]
                res = find_clique(new_remaining, new_current, size-1)
                if res:
                    return res
            return None
            
        all_nodes = list(range(n_inv))
        clique = find_clique(all_nodes, [], 5)
        
        if clique:
            # Verify 3-cycle condition: g_i g_j g_k != 1
            valid = True
            for i in range(5):
                for j in range(i+1, 5):
                    for kk in range(j+1, 5):
                        u = invol_g[clique[i]]
                        v = invol_g[clique[j]]
                        w = invol_g[clique[kk]]
                        uv = mul_g(u, v)
                        uvw = mul_g(uv, w)
                        if uvw == (0,0,0,0):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
            if valid:
                gen_set = [invol_g[idx] for idx in clique]
                
                # Construct graph
                all_elements = []
                for a in range(6):
                    for b in range(6):
                        for c in range(6):
                            for d in range(6):
                                all_elements.append((a,b,c,d))
                                
                elem_to_idx = {e: i for i, e in enumerate(all_elements)}
                
                edges = set()
                for u in all_elements:
                    idx_u = elem_to_idx[u]
                    for v in gen_set:
                        w = mul_g(u, v)
                        idx_w = elem_to_idx[w]
                        if idx_u < idx_w:
                            edges.add((idx_u, idx_w))
                        else:
                            edges.add((idx_w, idx_u))
                            
                n = len(all_elements)
                return n, list(edges)

        return None

    def luw_construct(k, g, d):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k."""
        q = k
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

    # 1. Try Cayley graph on S_3^4
    cayley_result = try_cayley_s3_4(k, g, time_limit / 2)
    if cayley_result is not None:
        return cayley_result
    
    # 2. Try L-U-W with d=4
    n4, edges4 = luw_construct(k, g, 4)
    if verify_girth(n4, edges4, g):
        return n4, edges4
        
    # 3. Try L-U-W with d=5
    n5, edges5 = luw_construct(k, g, 5)
    return n5, edges5
# EVOLVE-BLOCK-END
