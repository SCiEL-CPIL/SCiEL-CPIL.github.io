import time

def construct():
    P = 27720  # LCM(1..12)
    max_L = 144
    best_m = 0
    best_f = 0
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    def has_matching(m, L):
        # Build adjacency masks
        adj = [0] * 12
        adj_deg = [0] * 12
        for i in range(1, 13):
            d = i
            start = (-(m + 1)) % d
            mask = 0
            count = 0
            j = start
            while j < L:
                mask |= (1 << j)
                count += 1
                j += d
            adj[i - 1] = mask
            adj_deg[i - 1] = count
        
        # Quick check: if any left node has degree 0, no matching
        for i in range(12):
            if adj_deg[i] == 0:
                return False
        
        # Hopcroft-Karp algorithm
        # matchL[v] = matched right node for left node v, or -1
        # matchR[u] = matched left node for right node u, or -1
        matchL = [-1] * 12
        matchR = [-1] * L
        dist = [0] * 12
        
        from collections import deque
        
        def bfs():
            queue = deque()
            for v in range(12):
                if matchL[v] == -1:
                    dist[v] = 0
                    queue.append(v)
                else:
                    dist[v] = -1
            found = False
            while queue:
                v = queue.popleft()
                mask = adj[v]
                m_mask = mask
                while m_mask:
                    lsb = m_mask & (-m_mask)
                    u = lsb.bit_length() - 1
                    m_mask ^= lsb
                    if matchR[u] != -1:
                        w = matchR[u]
                        if dist[w] == -1:
                            dist[w] = dist[v] + 1
                            queue.append(w)
                    else:
                        found = True
            return found
        
        def dfs(v):
            mask = adj[v]
            m_mask = mask
            while m_mask:
                lsb = m_mask & (-m_mask)
                u = lsb.bit_length() - 1
                m_mask ^= lsb
                if matchR[u] == -1:
                    matchL[v] = u
                    matchR[u] = v
                    return True
                w = matchR[u]
                if dist[w] == dist[v] + 1:
                    if dfs(w):
                        matchL[v] = u
                        matchR[u] = v
                        return True
            dist[v] = -1
            return False
        
        matching = 0
        while bfs():
            for v in range(12):
                if matchL[v] == -1:
                    if dfs(v):
                        matching += 1
                        if matching == 12:
                            return True
        return matching == 12

    def compute_f(m):
        # Binary search for minimum L where matching exists
        lo, hi = 12, max_L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # First pass: scan all m in the period
    for m in range(P):
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    # Second pass: if we found a good f, try to verify and potentially improve
    # by checking nearby values more carefully
    if best_f > 0 and time.time() < deadline - 5:
        # Try a few more iterations around the best
        for offset in range(-50, 51):
            m_candidate = (best_m + offset) % P
            if m_candidate < 0:
                m_candidate += P
            if m_candidate == best_m:
                continue
            if time.time() > deadline - 2:
                break
            f_val = compute_f(m_candidate)
            if f_val > best_f:
                best_f = f_val
                best_m = m_candidate

    return best_m
