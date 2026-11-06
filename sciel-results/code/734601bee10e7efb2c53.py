import random
import time

def _is_strongly_connected(n, orientation):
    adj = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i].append(j)
            else:
                adj[j].append(i)
            idx += 1
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    if count < n:
        return False
    radj = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                radj[j].append(i)
            else:
                radj[i].append(j)
            idx += 1
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in radj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == n

def _count_bad_quads(n, orientation):
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    bad = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    if not (adj[a][b] and adj[a][c] and adj[a][d]):
                        if not (adj[b][a] and adj[b][c] and adj[b][d]):
                            if not (adj[c][a] and adj[c][b] and adj[c][d]):
                                if not (adj[d][a] and adj[d][b] and adj[d][c]):
                                    bad += 1
    return bad

def construct():
    # Try to find a valid strongly connected tournament for small n
    # If the problem doesn't require SC, n=5 is valid.
    # Given the hint "smallest known 67", SC is likely required.
    
    # For n=5, check all SC tournaments? There are few.
    # We'll use a heuristic search for n from 5 to 60.
    
    for n in range(5, 60):
        # Try a few random restarts
        for _ in range(50):
            orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            # Ensure SC? Hard to generate directly.
            # Just check validity. If valid and SC, return.
            if _is_strongly_connected(n, orientation) and _count_bad_quads(n, orientation) == 0:
                return {'n': n, 'orientation': orientation}
                
    # If not found, try 67 with more effort
    # Fallback to Paley 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
