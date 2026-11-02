import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2 - 1
            return k * ((k-1)**h - 1) // (k - 2) + 1
        else:
            h = (g - 1) // 2
            return 1 + k * ((k-1)**h - 1) // (k - 2)
    
    mb = moore_bound(k, g)
    
    # LUW construction as fallback
    def luw_construct(k, g):
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
                
        luw_n = len(order)
        luw_edges = edges
        return luw_n, luw_edges
    
    luw_n, luw_edges = luw_construct(k, g)
    
    # Check girth function
    def check_girth(n, adj, g):
        for s in range(n):
            dist = {}
            q2 = deque()
            for v in adj[s]:
                q2.append((v, s))
                dist[(v, s)] = 1
            while q2:
                cur, prev = q2.popleft()
                d_cur = dist[(cur, prev)]
                if d_cur >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == s:
                        if d_cur + 1 < g:
                            return False
                    else:
                        if (nxt, cur) not in dist:
                            dist[(nxt, cur)] = d_cur + 1
                            q2.append((nxt, cur))
        return True
    
    # Verify LUW
    adj = [[] for _ in range(luw_n)]
    for u, v in luw_edges:
        adj[u].append(v)
        adj[v].append(u)
    luw_ok = check_girth(luw_n, adj, g)
    
    if not luw_ok:
        return luw_n, luw_edges
    
    # Try to find smaller graph via backtracking
    # For k=4, g=10, try n from mb to min(luw_n-1, 400)
    best_n = luw_n
    best_edges = luw_edges
    
    if luw_n > mb:
        n_max_search = min(luw_n - 1, mb + 200)
        
        for n in range(mb, n_max_search + 1):
            if time.time() > deadline - 5:
                break
            
            # Try random construction with girth verification
            found = False
            trials = 50
            
            for trial in range(trials):
                if time.time() > deadline - 5:
                    break
                
                # Generate random 4-regular graph
                # Use configuration model
                random.seed(trial * n + 42)
                
                # Build degree sequence
                deg_seq = [k] * n
                
                # Configuration model: create stubs
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                
                random.shuffle(stubs)
                
                # Pair up stubs to form edges
                edges_set = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if u > v:
                        u, v = v, u
                    if (u, v) in edges_set:
                        valid = False
                        break
                    edges_set.add((u, v))
                
                if not valid:
                    continue
                
                # Build adjacency
                adj2 = [[] for _ in range(n)]
                for u, v in edges_set:
                    adj2[u].append(v)
                    adj2[v].append(u)
                
                # Check regular
                if any(len(adj2[v]) != k for v in range(n)):
                    continue
                
                # Check girth
                if check_girth(n, adj2, g):
                    best_n = n
                    best_edges = list(edges_set)
                    found = True
                    break
            
            if found:
                break
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
