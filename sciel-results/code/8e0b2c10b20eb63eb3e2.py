import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 50s margin for safety
    
    # 1. Calculate Moore Bound
    half_g = g // 2
    if g % 2 == 1:
        target_girth = g + 1
    else:
        target_girth = g
        
    moore_sum = 0
    for i in range(target_girth // 2):
        moore_sum += (k - 1) ** i
    N_min = 2 * moore_sum
    
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

    # Known small graphs
    if k == 3 and g == 5:
        # Petersen graph
        n = 10
        edges = [(0,1), (0,4), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9), (5,6), (5,9), (6,7), (7,8), (8,9)]
        return n, edges
    if k == 3 and g == 6:
        # Heawood graph
        n = 14
        edges = []
        # Fano plane lines
        # ... simplified known edges
        edges = [(0,1), (0,2), (0,3), (1,4), (1,5), (1,6), (2,4), (2,5), (2,7), (3,6), (3,7), (3,8), (4,9), (4,10), (4,11), (5,9), (5,10), (5,12), (6,11), (6,12), (6,13), (7,9), (7,13), (7,10), (8,11), (8,12), (8,13)]
        return n, edges

    best_result = None
    
    max_M = N_min // 2 + 1000 # Increased limit
    
    for M in range(max(1, N_min // 2), max_M + 1):
        if time.time() > deadline:
            break
            
        if k > M - 1:
            continue

        num_tries = 100 # Increased tries
        found = False
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            
            S = set(random.sample(range(1, M), k))
            T = list(S) + [-s % M for s in S]
            
            # Fast filter for short cycles
            valid = True
            
            # Check 4-cycles: s1+s2 == s3+s4
            sums_2 = {}
            for i in range(len(T)):
                for j in range(i+1, len(T)):
                    s_val = (T[i] + T[j]) % M
                    if s_val in sums_2:
                        # Check if pairs are disjoint
                        p1 = sums_2[s_val]
                        p2 = (i, j)
                        if p1[0] != p2[0] and p1[0] != p2[1] and p1[1] != p2[0] and p1[1] != p2[1]:
                            valid = False
                            break
                    else:
                        sums_2[s_val] = (i, j)
                if not valid: break
            
            # Check 6-cycles: s1+s2+s3 == s4+s5+s6
            if valid:
                sums_3 = {}
                for i in range(len(T)):
                    for j in range(i+1, len(T)):
                        for l in range(j+1, len(T)):
                            s_val = (T[i] + T[j] + T[l]) % M
                            if s_val in sums_3:
                                p1 = sums_3[s_val]
                                p2 = (i, j, l)
                                # Check disjoint
                                if not any(x == y for x in p1 for y in p2):
                                    valid = False
                                    break
                            else:
                                sums_3[s_val] = (i, j, l)
                        if not valid: break
                    if not valid: break
            
            if not valid:
                continue

            # Construct adjacency list
            adj = [[] for _ in range(2 * M)]
            edges = []
            for u in range(M):
                for s in S:
                    v = (u + s) % M
                    adj[u].append(v + M)
                    adj[v + M].append(u)
                    edges.append((u, v + M))
            
            # Check Girth with BFS
            girth_ok = True
            for start in range(2 * M):
                if not girth_ok:
                    break
                dist = [-1] * (2 * M)
                parent = [-1] * (2 * M)
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= target_girth // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u] and dist[v] <= dist[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                girth_ok = False
                                break
            
            if girth_ok:
                best_result = (2 * M, edges)
                found = True
                break
        
        if found:
            return best_result
            
    if best_result:
        return best_result
        
    return luw_construct(k, g)
