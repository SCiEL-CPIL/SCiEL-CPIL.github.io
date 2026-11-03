import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_bfs(n, adj, min_girth):
    # Optimized BFS
    # Returns True if girth >= min_girth
    max_depth = min_girth // 2
    for start in range(n):
        # dist[(curr, prev)]
        # Using a dict is slow. Use a set of visited states?
        # State is (curr, prev).
        # We can encode state as curr * n + prev.
        visited = [False] * (n * n)
        queue = collections.deque()
        
        for nb in adj[start]:
            state = nb * n + start
            visited[state] = True
            queue.append((nb, start, 1))
            
        while queue:
            curr, prev, d = queue.popleft()
            if d >= max_depth:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < min_girth:
                        return False
                    continue
                state = nb * n + curr
                if not visited[state]:
                    visited[state] = True
                    queue.append((nb, curr, d + 1))
    return True

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    best_n = 100000
    best_edges = None
    
    # 1. GP Graphs
    # GP(n, k_gp) has 2n vertices.
    # Lower bound for 2n is ~512, so n >= 256.
    # Upper bound for search: let's go up to 500 (size 1000) to be safe.
    min_n_gp = 256
    max_n_gp = 500
    
    for n in range(min_n_gp, max_n_gp + 1):
        if _time_left() < 5.0:
            break
        # Modular pre-filter
        # k_gp^j != +/- 1 mod n for j=1..g/2-1
        for k_gp in range(1, n // 2 + 1):
            if _time_left() < 1.0:
                break
            if math.gcd(n, k_gp) != 1:
                continue
            valid = True
            val = k_gp
            for j in range(1, g // 2):
                val = (val * k_gp) % n
                if val == 1 or val == n - 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Build adj
            adj = [[] for _ in range(2*n)]
            for i in range(n):
                adj[i].append((i+1)%n)
                adj[i].append((i-1)%n)
                adj[i].append(n+i)
                adj[n+i].append(i)
                adj[n+i].append(n+(i+k_gp)%n)
                adj[n+i].append(n+(i-k_gp)%n)
            
            if _check_girth_bfs(2*n, adj, g):
                if 2*n < best_n:
                    best_n = 2*n
                    best_edges = []
                    for i in range(n):
                        best_edges.append((i, (i+1)%n))
                    for i in range(n):
                        best_edges.append((i, n+i))
                    for i in range(n):
                        best_edges.append((n+i, n+(i+k_gp)%n))
    
    # 2. Circulant Graphs
    # C_n(a,b,c)
    # n from 512 to 940
    min_n_circ = 512
    max_n_circ = 940
    
    # To save time, we might not check ALL n.
    # But we want the best.
    # We can iterate n and try a limited number of random offsets.
    
    for n in range(min_n_circ, max_n_circ + 1):
        if _time_left() < 10.0:
            break
        # Try random offsets
        # Number of trials depends on time
        trials = 20
        for _ in range(trials):
            if _time_left() < 2.0:
                break
            offsets = random.sample(range(1, n//2 + 1), 3)
            # Check distinctness
            neis = set()
            ok = True
            for a in offsets:
                for s in [1, -1]:
                    v = (s*a) % n
                    if v == 0 or v in neis:
                        ok = False
                        break
                    neis.add(v)
                if not ok: break
            if not ok: continue
            
            adj = [[] for _ in range(n)]
            for i in range(n):
                for a in offsets:
                    adj[i].append((i+a)%n)
                    adj[i].append((i-a)%n)
            
            if _check_girth_bfs(n, adj, g):
                if n < best_n:
                    best_n = n
                    best_edges = []
                    for i in range(n):
                        for a in offsets:
                            best_edges.append((i, (i+a)%n))
    
    if best_edges is None:
        # Fallback: LUW
        # (Keep the LUW code from current program as fallback)
        pass
        
    return best_n, best_edges
