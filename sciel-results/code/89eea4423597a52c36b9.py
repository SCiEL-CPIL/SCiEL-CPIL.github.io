"""Randomized Local Search (RLS) for high-girth regular graphs.

Approach:
1. Start with random k-regular graphs of increasing size n (starting from Moore bound).
2. Use a local search heuristic to increase the girth:
   - Find the shortest cycle.
   - If girth < target, attempt to break the shortest cycle via edge swaps (2-opt).
   - Accept swaps that preserve regularity/simplicity and do not decrease girth.
3. This is distinct from the "greedy growth" approach of the seed program.
"""
import random
import sys
import time
from collections import deque

def _girth_nonbacktracking(n, adj):
    """Exact girth via non-backtracking BFS from every vertex."""
    best = 10**9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq.append((s, nb, 1))
        
        while dq:
            v, prev, d = dq.popleft()
            if d >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    cyc = d + 1
                    if cyc < best:
                        best = cyc
                    continue
                st = (v, nb)
                if st in dist:
                    cyc = d + dist[st]
                    if cyc < best:
                        best = cyc
                else:
                    dist[st] = d + 1
                    dq.append((v, nb, d + 1))
    return best

def _random_k_regular(n, k, rng):
    """Generate a simple random k-regular graph via configuration model with retries."""
    if n * k % 2 != 0:
        return None
    for _ in range(50):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                valid = False
                break
            if b in adj[a]:
                valid = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if valid:
            if all(len(adj[i]) == k for i in range(n)):
                return adj
    return None

def _find_shortest_cycle(n, adj, max_len):
    """Find a shortest cycle of length < max_len. Returns list of vertices in cycle or None."""
    best_cycle = None
    best_len = max_len
    for s in range(n):
        visited = {}
        dq = deque()
        for nb in adj[s]:
            visited[(s, nb)] = 1
            dq.append((s, nb, [s, nb]))
        
        found = False
        while dq:
            v, prev, path = dq.popleft()
            d = len(path) - 1
            if d >= best_len:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    cyc_len = d + 1
                    if cyc_len < best_len:
                        best_len = cyc_len
                        best_cycle = path + [s]
                        found = True
                    continue
                st = (v, nb)
                if st not in visited:
                    visited[st] = d + 1
                    dq.append((v, nb, path + [nb]))
        if found:
            break
    return best_cycle

def _try_swap(adj, e1, e2):
    """Try to swap edges e1=(u,v) and e2=(x,y) to (u,x) and (v,y)."""
    u, v = e1
    x, y = e2
    
    if v not in adj[u] or u not in adj[v]:
        return False
    if y not in adj[x] or x not in adj[y]:
        return False
        
    if u == x or v == y or u == y or v == x:
        return False
        
    if x in adj[u] or y in adj[v]:
        return False
        
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    
    return True

def _local_search(adj, n, k, g, max_iters=500):
    """Improve girth using local search."""
    current_girth = _girth_nonbacktracking(n, adj)
    if current_girth >= g:
        return True
    
    for _ in range(max_iters):
        cycle = _find_shortest_cycle(n, adj, g)
        if cycle is None:
            return True
        
        cycle_edges = []
        m = len(cycle)
        for i in range(m):
            u = cycle[i]
            v = cycle[(i+1)%m]
            cycle_edges.append((u, v))
            
        e1 = random.choice(cycle_edges)
        
        x = random.randrange(n)
        if not adj[x]:
            continue
        y = random.choice(adj[x])
        e2 = (x, y)
        
        adj_backup = [list(nei) for nei in adj]
        
        if _try_swap(adj, e1, e2):
            new_girth = _girth_nonbacktracking(n, adj)
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= g:
                    return True
            else:
                adj[:] = adj_backup
    return False

def construct(k, g):
    start_time = time.time()
    time_limit = 115
    
    n_min = 512
    n_max = 1000
    
    for n in range(n_min, n_max + 1, 2):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 20
        
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            rng = random.Random()
            adj = _random_k_regular(n, k, rng)
            if adj is None:
                continue
            
            girth = _girth_nonbacktracking(n, adj)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            
            success = _local_search(adj, n, k, g, max_iters=100)
            if success:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback
    n_luw, edges_luw = _luw_fallback(k, g)
    return n_luw, edges_luw

def _luw_fallback(k, g):
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
