import time
import sys

sys.setrecursionlimit(10000)

P = 27720

def _max_matching_size(r, L):
    """Return the maximum matching size for the bipartite graph defined by r and L.
    r[i] = (-base) % (i+1) for i in 0..11, so label (i+1) can match positions j where j ≡ r[i] (mod i+1).
    """
    # Build adjacency: for each label li (0..11), which positions j in [0, L) are available?
    # Position j corresponds to value base + j, which is divisible by (li+1) iff j ≡ r[li] (mod li+1)
    adj = []
    for li in range(12):
        step = li + 1
        start = r[li]
        if start >= L:
            adj.append([])
        else:
            adj.append(list(range(start, L, step)))
    
    # Hopcroft-Karp or simple DFS augmenting path
    # For small graphs (12 nodes on left, up to 144 on right), simple DFS is fine
    match_pos = [-1] * L  # match_pos[j] = label index assigned to position j, or -1
    
    def dfs(li, visited):
        for j in adj[li]:
            if visited[j]:
                continue
            visited[j] = True
            if match_pos[j] == -1 or dfs(match_pos[j], visited):
                match_pos[j] = li
                return True
        return False
    
    size = 0
    for li in range(12):
        visited = [False] * L
        if dfs(li, visited):
            size += 1
    
    return size

def _f_for_residue(r):
    """Given the residue vector r for a specific m mod P, find the minimum L such that
    a perfect matching exists in (m, m+L]."""
    lo, hi = 12, 144
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _max_matching_size(r, mid) == 12:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    best_m = 0
    best_f = 0
    
    for m in range(P):
        if time.time() > deadline:
            break
        base = m + 1
        r = [(-base) % (i + 1) for i in range(12)]
        f = _f_for_residue(r)
        if f > best_f:
            best_f = f
            best_m = m
    
    return int(best_m)
