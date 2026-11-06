"""Search for 4-coloring of K_n with no monochromatic C6. Maximize n."""

import time
import random
from collections import deque


def _edge_idx(i, j, n):
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _makes_c6(colors, n, i, j, c):
    adj = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if a == i and b == j:
                continue
            idx = _edge_idx(a, b, n)
            if colors[idx] == c:
                adj[a].append(b)
                adj[b].append(a)
    dist = [-1] * n
    dist[i] = 0
    q = deque([i])
    while q:
        u = q.popleft()
        if dist[u] >= 5:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if v == j and dist[v] == 5:
                    return True
                if dist[v] < 5:
                    q.append(v)
    return dist[j] == 5


def _valid(colors, n):
    for c in range(4):
        adj = [[] for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                idx = _edge_idx(a, b, n)
                if colors[idx] == c:
                    adj[a].append(b)
                    adj[b].append(a)
        for a in range(n):
            for b in range(a + 1, n):
                idx = _edge_idx(a, b, n)
                if colors[idx] != c:
                    continue
                adj[a].remove(b)
                adj[b].remove(a)
                dist = [-1] * n
                dist[a] = 0
                q = deque([a])
                while q:
                    u = q.popleft()
                    if dist[u] >= 5:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if v == b and dist[v] == 5:
                                adj[a].append(b)
                                adj[b].append(a)
                                return False
                            if dist[v] < 5:
                                q.append(v)
                if dist[b] == 5:
                    adj[a].append(b)
                    adj[b].append(a)
                    return False
                adj[a].append(b)
                adj[b].append(a)
    return True


def _greedy(n, deadline, max_attempts=500):
    total = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        rng = random.Random(attempt * 7919 + n * 31)
        colors = [-1] * total
        order = list(range(total))
        rng.shuffle(order)
        ok = True
        for pos in order:
            i, j = edges[pos]
            cols = [0, 1, 2, 3]
            rng.shuffle(cols)
            placed = False
            for c in cols:
                colors[pos] = c
                if not _makes_c6(colors, n, i, j, c):
                    placed = True
                    break
                colors[pos] = -1
            if not placed:
                ok = False
                break
        if ok and _valid(colors, n):
            return colors
    return None


def _backtrack(n, deadline):
    total = n * (n - 1) // 2
    # Strategic edge order: group by smaller vertex
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    # This is already in lexicographic order which is good
    
    colors = [-1] * total
    colors[0] = 0  # (0,1)
    colors[1] = 0  # (0,2)
    
    for c12 in [0, 1]:
        for c03 in [0, 1, 2]:
            colors[3] = c12  # (1,2)
            colors[2] = c03  # (0,3)
            if _makes_c6(colors, n, 1, 2, c12) or _makes_c6(colors, n, 0, 3, c03):
                colors[2] = -1
                colors[3] = -1
                continue
            
            def bt(pos):
                if time.time() > deadline:
                    return False
                if pos >= total:
                    return True
                i, j = edges[pos]
                # Try colors, prefer those that create fewer C6 risks
                for c in range(4):
                    colors[pos] = c
                    if not _makes_c6(colors, n, i, j, c):
                        if bt(pos + 1):
                            return True
                    colors[pos] = -1
                return False
            
            if bt(4):
                return colors.copy()
            colors[2] = -1
            colors[3] = -1
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    best_n = 0
    best_colors = None
    
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
        colors = _greedy(n, deadline, max_attempts=200)
        if colors is not None:
            best_n = n
            best_colors = colors
            break
        if n <= 14:
            colors = _backtrack(n, deadline)
            if colors is not None:
                best_n = n
                best_colors = colors
                break
    
    if best_n == 0:
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    result = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            result.append(best_colors[_edge_idx(i, j, best_n)])
    return {'n': best_n, 'colors': result}
