from collections import deque

def construct(k, g):
    if k != 5 or g != 9:
        raise ValueError("Only (5,9) supported")
    
    def check(n, a, b, min_g):
        half = n // 2
        dist = [-1] * n
        par = [-1] * n
        dist[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n, (u + half) % n):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                elif v != par[u] and dist[u] + dist[v] + 1 < min_g:
                    return False
        return True
    
    for n in range(428, 800, 2):
        half = n // 2
        # Try a = n//5, b = 2*n//5 type values first
        a1 = n // 5
        b1 = 2 * n // 5
        if check(n, a1, b1, g):
            edges = [(v, (v + a1) % n) for v in range(n) if v < (v + a1) % n]
            edges += [(v, (v + b1) % n) for v in range(n) if v < (v + b1) % n]
            edges += [(v, (v + half) % n) for v in range(n) if v < (v + half) % n]
            return n, edges
        # Try a few more
        for a in range(max(1, n // 6), n // 3):
            for b in range(a + 1, 2 * n // 3):
                if b == half:
                    continue
                if check(n, a, b, g):
                    edges = [(v, (v + a) % n) for v in range(n) if v < (v + a) % n]
                    edges += [(v, (v + b) % n) for v in range(n) if v < (v + b) % n]
                    edges += [(v, (v + half) % n) for v in range(n) if v < (v + half) % n]
                    return n, edges
    raise ValueError("No solution")
