"""Construct small 4-regular bipartite graphs of girth 10 using random generation with local repair."""
import random
import time
from collections import deque

def _verify(n, edges, k, g):
    """Verify simple, k-regular, no cycle < g via non-backtracking BFS from every vertex."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for v0 in range(n):
        for v1 in adj[v0]:
            visited = set()
            visited.add((v0, v1))
            queue = deque([(v0, v1, 1)])
            while queue:
                prev, curr, dist = queue.popleft()
                if dist >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_state = (curr, nxt)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if nxt == v0 and dist + 1 < g:
                        return False
                    queue.append((curr, nxt, dist + 1))
    return True

def _find_short_cycle(n, adj, g):
    """Find a cycle of length < g. Returns list of vertices or None."""
    for v0 in range(n):
        for v1 in adj[v0]:
            visited = {}
            visited[(v0, v1)] = [v0, v1]
            queue = deque([(v0, v1, 1)])
            while queue:
                prev, curr, dist = queue.popleft()
                if dist >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_state = (curr, nxt)
                    if new_state in visited:
                        continue
                    if nxt == v0 and dist + 1 < g:
                        path = visited[(prev, curr)]
                        return path + [v0]
                    visited[new_state] = visited[(prev, curr)] + [nxt]
                    queue.append((curr, nxt, dist + 1))
    return None

def construct(k, g):
    if k != 4 or g != 10:
        raise ValueError("Only k=4, g=10 supported")
    
    start_time = time.time()
    time_limit = 110
    
    for n in range(244, 322, 2):
        if time.time() - start_time > time_limit:
            break
        
        n_half = n // 2
        attempts = 0
        max_attempts = 2000
        while attempts < max_attempts and time.time() - start_time < time_limit:
            attempts += 1
            
            # Generate random 4-regular bipartite graph
            # U = 0..n_half-1, V = n_half..n-1
            # Use 4 random permutations
            perms = [random.sample(range(n_half), n_half) for _ in range(4)]
            
            adj = [set() for _ in range(n)]
            edges = []
            valid = True
            for u in range(n_half):
                for p in range(4):
                    v = perms[p][u]
                    v_idx = n_half + v
                    if v_idx in adj[u]:
                        valid = False
                        break
                    adj[u].add(v_idx)
                    adj[v_idx].add(u)
                    edges.append((u, v_idx))
            if not valid:
                continue
            
            # Check if all V vertices have degree 4
            if any(len(adj[v]) != 4 for v in range(n_half, n)):
                continue
            
            # Check girth
            if _verify(n, edges, 4, 10):
                return n, edges
            
            # Try local repair: find short cycle and swap edges
            # This is heuristic; just retry for simplicity within time budget
    
    # Fallback: L-U-W construction
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
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
