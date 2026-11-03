import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
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

    # Fallback result
    seed_n, seed_edges = seed_construct(k, g)
    
    # Try to find a better graph using PSL(2, 13) for k=5, g=9
    if k == 5 and g == 9:
        try:
            q = 13
            # Generate PSL(2, 13) elements
            # Represent as (a, b, c, d) with ad-bc=1
            # Canonical form: if a != 0, keep. If a==0 and b!=0, keep. If a==0,b==0, c!=0, make c positive?
            # Simpler: Map M to M if "first" non-zero entry in [a,b,c,d] is positive, else -M.
            
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # ad - bc = 1
                        if a != 0:
                            d_val = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d_val))
                        else:
                            # -bc = 1 => bc = -1
                            if b == 0:
                                continue
                            if (-b * c) % q != 1:
                                continue
                            for d_val in range(q):
                                sl_elements.append((a, b, c, d_val))
            
            # Reduce to PSL
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d_val = e
                key = (a, b, c, d_val)
                neg_key = (-a % q, -b % q, -c % q, -d_val % q)
                # Canonical: first non-zero in tuple is positive
                def is_canon(t):
                    for x in t:
                        if x != 0:
                            return x > 0
                    return False
                if is_canon(key):
                    rep = key
                else:
                    rep = neg_key
                if rep not in seen:
                    seen.add(rep)
                    psl_elements.append(rep)
            
            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            
            # Multiplication
            def mul_mat(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                return ((a1 * a2 + b1 * c2) % q,
                        (a1 * b2 + b1 * d2) % q,
                        (c1 * a2 + d1 * c2) % q,
                        (c1 * b2 + d1 * d2) % q)
            
            # Precompute multiplication table? 1092^2 = 1.2M entries.
            # Memory: 1.2M * 4 bytes = 5 MB. OK.
            # Time: 1.2M * 8 mults = 10M ops. Fast.
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            for i in range(n_vertices):
                m1 = psl_elements[i]
                for j in range(n_vertices):
                    m2 = psl_elements[j]
                    prod = mul_mat(m1, m2)
                    # prod might not be canonical
                    a, b, c, d_val = prod
                    if a != 0 or (a==0 and b != 0):
                         # Check if canonical
                         # If not, negate
                         pass # Just use idx_map which has canonical keys
                    # idx_map keys are canonical.
                    # prod might be non-canonical.
                    # Find canonical rep of prod
                    def get_canon(t):
                        a,b,c,d = t
                        if a != 0:
                             if a > 0: return t
                             else: return (-a%q, -b%q, -c%q, -d%q)
                        elif b != 0:
                             if b > 0: return t
                             else: return (-a%q, -b%q, -c%q, -d%q)
                        elif c != 0:
                             if c > 0: return t
                             else: return (-a%q, -b%q, -c%q, -d%q)
                        else:
                             if d > 0: return t
                             else: return (-a%q, -b%q, -c%q, -d%q)
                    prod_canon = get_canon(prod)
                    mul_table[i, j] = idx_map[prod_canon]
            
            # Find involutions
            involution_indices = []
            identity = idx_map[(1, 0, 0, 1)]
            for i in range(n_vertices):
                if i == identity: continue
                if mul_table[i, i] == identity:
                    involution_indices.append(i)
            
            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)
            
            # Search for 5 involutions
            found = False
            best_edges = None
            
            # Random search
            # We want to avoid short cycles.
            # Heuristic: Pick involutions that are "far" apart?
            # Or just random.
            
            max_attempts = 5000
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                    
                # Pick 5 distinct involutions
                S_idx = np.random.choice(n_inv, size=5, replace=False)
                
                # Check generation
                visited = np.zeros(n_vertices, dtype=bool)
                visited[identity] = True
                queue = [identity]
                head = 0
                # edges list for this candidate
                cand_edges = []
                
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    for s_idx in S_idx:
                        v = mul_table[u, s_idx]
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            cand_edges.append((u, v))
                
                if not visited.all():
                    continue
                
                # Check girth
                # BFS from identity
                dist = np.full(n_vertices, -1, dtype=np.int32)
                parent = np.full(n_vertices, -1, dtype=np.int32)
                dist[identity] = 0
                queue_nb = [identity]
                head_nb = 0
                girth_ok = True
                
                while head_nb < len(queue_nb):
                    u = queue_nb[head_nb]
                    head_nb += 1
                    if dist[u] >= 4: # Max depth for girth 9 check (cycles up to 8)
                        continue
                    for s_idx in S_idx:
                        v = mul_table[u, s_idx]
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue_nb.append(v)
                        else:
                            # Found a cycle
                            if v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    girth_ok = False
                                    break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # Verify degree and simplicity
                    # cand_edges has n_vertices * 5 / 2 edges?
                    # Since S contains involutions, (u, v) and (v, u) are same.
                    # We added (u, v) only when v was unvisited.
                    # This generates a spanning tree + back edges?
                    # No, BFS generates all edges?
                    # In the generation BFS, we added (u, v) for every edge traversed.
                    # Since the graph is undirected and we traverse both ways?
                    # No, we traverse u->v. If v is unvisited, add edge.
                    # This adds each edge exactly once (from the side that discovers the node).
                    # So cand_edges contains all edges of the graph.
                    
                    # Check if any edge is a loop (u==v)
                    # Involutions are not identity, so u != v.
                    # Check for repeated edges?
                    # BFS ensures each edge is added once.
                    
                    found = True
                    best_edges = cand_edges
                    break
            
            if found:
                return n_vertices, best_edges
                
        except Exception as e:
            pass # Fall back to seed

    return seed_n, seed_edges
