import time

N = 12
L_MAX = 150
PERIOD = 27720

# Precompute j % i for j in 1..L_MAX, i in 1..N
# j_mod[i][j] = j % i
j_mod = [[0] * (L_MAX + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, L_MAX + 1):
        j_mod[i][j] = j % i

def check_match(m, L):
    # m_mod[i] = m % i
    m_mod = [0] * (N + 1)
    for i in range(1, N + 1):
        m_mod[i] = m % i
        
    # Build adjacency lists
    # adj[i] is list of j (1..L)
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        adj_i = adj[i]
        jm = j_mod[i]
        mm = m_mod[i]
        for j in range(1, L + 1):
            if (mm + jm[j]) % i == 0:
                adj_i.append(j)
                
    # DFS Bipartite Matching
    match_to = [0] * (L + 1)
    
    def dfs(i, seen):
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_to[j] == 0 or dfs(match_to[j], seen):
                match_to[j] = i
                return True
        return False

    count = 0
    for i in range(1, N + 1):
        seen = [False] * (L + 1)
        if dfs(i, seen):
            count += 1
            if count == N:
                return True
    return count == N

def compute_f(m):
    lo, hi = 1, L_MAX
    ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if check_match(m, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    deadline = time.time() + 55  # 55s limit
    best_m = 0
    best_f = 0
    
    # We scan the full period.
    # If it's too slow, we might need to stride.
    # Let's try full scan first.
    for m in range(PERIOD):
        if time.time() > deadline:
            break
        f = compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    return int(best_m)
