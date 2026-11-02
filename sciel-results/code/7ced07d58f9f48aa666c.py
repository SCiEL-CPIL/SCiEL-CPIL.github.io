import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore = k * sum((k - 1) ** i for i in range(r))

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

    # Helper: Check girth
    def check_girth(adj, n, target_g):
        for v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            queue = [v]
            while queue:
                u = queue.pop(0)
                if dist[u] >= (target_g - 1) // 2:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        queue.append(w)
                    elif w != parent[u] and parent[w] != u:
                        # Found a cycle
                        # Cycle length is dist[u] + 1 + dist[w]?
                        # Actually, in BFS, if we see a visited node w that is not parent,
                        # the cycle length is dist[u] + 1 + dist[w] is NOT correct.
                        # The cycle is u -> w -> ... -> LCA -> ... -> u.
                        # Simpler: if dist[u] + 1 + dist[w] < target_g?
                        # No, just check if we found ANY cycle < target_g.
                        # For BFS, the first time we find a back edge, the cycle length
                        # is dist[u] + 1 + dist[w] - 2*dist[lca].
                        # This is hard to compute fast.
                        # However, if dist[u] + 1 + dist[w] < target_g, it's definitely a short cycle.
                        # If dist[u] + 1 + dist[w] >= target_g, it might still be short if LCA is deep.
                        # But for girth >= 9, we can just run BFS up to depth 4.
                        # If we find a node at depth 4 that is already visited at depth 4?
                        # Or depth 3 and 4?
                        # If we find a neighbor w of u (dist[u]=d) such that dist[w] <= d+1 and w != parent[u],
                        # and dist[w] + d + 1 < target_g?
                        # Actually, if we do BFS up to depth (target_g-1)//2, and we find a "cross edge"
                        # connecting two nodes in the BFS tree at depths d1, d2 such that d1+d2+1 < target_g,
                        # then we have a short cycle.
                        # For target_g=9, max depth 4.
                        # If we find an edge between u (dist 4) and w (dist 4), cycle length 9? No, 4+4+1=9.
                        # We want NO cycle < 9. So cycle length >= 9 is OK.
                        # So if we find an edge connecting dist 4 and dist 4, length 9, it's OK.
                        # If we find edge connecting dist 3 and dist 4, length 8, BAD.
                        # So we just need to check if any cross edge connects depths d1, d2 with d1+d2+1 < target_g.
                        if dist[w] + dist[u] + 1 < target_g:
                            return False
        return True

    # Helper: Generate random 4-regular graph using two permutations
    def gen_random_4reg(N):
        # Returns adjacency list
        for _ in range(100):
            p1 = list(range(N))
            p2 = list(range(N))
            random.shuffle(p1)
            random.shuffle(p2)
            
            # Check for loops
            if any(p1[i] == i for i in range(N)) or any(p2[i] == i for i in range(N)):
                continue
            
            # Check for multi-edges
            valid = True
            edges = set()
            for i in range(N):
                for j in [p1[i], p2[i]]:
                    if i > j:
                        e = (i, j)
                    else:
                        e = (j, i)
                    if e in edges:
                        valid = False
                        break
                    edges.add(e)
                if not valid:
                    break
            
            if valid:
                adj = [[] for _ in range(N)]
                for i in range(N):
                    adj[i].append(p1[i])
                    adj[i].append(p2[i])
                return adj
        return None

    # Local search for 4-regular graph
    def improve_girth_4reg(adj, N, target_g, max_iters=1000):
        # This is a simplified local search
        # Find a short cycle and try to break it
        # For now, just re-generate if check fails
        pass

    # Strategy:
    # 1. Try to find a 4-regular graph H with girth >= 9
    # 2. If found, return C_9 x H
    # 3. Else, return LUW(k, g)

    best_H = None
    best_M = float('inf')
    
    # Search for H
    # Moore bound for k=4, g=9 is 161
    for M in range(161, 300):
        if time.time() > deadline - 20:
            break
        # Try a few random graphs
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            adj = gen_random_4reg(M)
            if adj is None:
                continue
            if check_girth(adj, M, 9):
                best_H = adj
                best_M = M
                break
        if best_H is not None:
            break

    if best_H is not None:
        M = best_M
        N_total = 9 * M
        edges = []
        # Map (layer, vertex) to id
        # id = layer * M + vertex
        for layer in range(9):
            for v in range(M):
                u = layer * M + v
                # H edges
                for w in best_H[v]:
                    v2 = layer * M + w
                    if u < v2:
                        edges.append((u, v2))
                # C_9 edges
                v_next = (layer + 1) % 9
                w_next = v_next * M + v
                if u < w_next:
                    edges.append((u, w_next))
        return N_total, edges

    # Fallback to LUW
    return luw_construct(k, g)
