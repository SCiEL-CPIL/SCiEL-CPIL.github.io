import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144

    # Precompute bitmasks: full_mask[i][rem] = bitmask of all d in 1..max_L where d % i == rem
    full_mask = {}
    for i in range(1, n + 1):
        full_mask[i] = [0] * i
        for d in range(1, max_L + 1):
            rem = d % i
            full_mask[i][rem] |= (1 << d)

    # Precompute popcount for masks up to 2^145 (use int.bit_count() in Python 3.10+)
    # We'll just use .bit_count() directly

    def has_perfect_matching(m, L):
        """Check if (m, m+L] has a perfect matching using Hopcroft-Karp."""
        if L < n:
            return False

        # Build adjacency: for each left node i (0-based), list of right nodes d (1..L)
        adj = [[] for _ in range(n)]
        union_mask = 0

        for i in range(1, n + 1):
            rem = (-m) % i
            mask = full_mask[i][rem] & ((1 << (L + 1)) - 1)
            union_mask |= mask
            # Extract set bits
            mm = mask
            while mm:
                d = (mm & (-mm)).bit_length() - 1
                adj[i - 1].append(d)
                mm &= mm - 1

        # Quick Hall checks
        for i in range(n):
            if not adj[i]:
                return False
        if union_mask.bit_count() < n:
            return False

        # Hopcroft-Karp
        # pairU[u] = v matched to u (0-based left), -1 if unmatched
        # pairV[v] = u matched to v (1-based right, indexed by d), -1 if unmatched
        pairU = [-1] * n
        pairV = [-1] * (L + 1)  # indexed by d from 1..L
        dist = [0] * n

        def bfs():
            """BFS to find shortest augmenting paths. Returns True if augmenting path exists."""
            queue = deque()
            for u in range(n):
                if pairU[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            """DFS to find augmenting path from u. Returns True if found."""
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = -1  # mark as dead
            return False

        matching = 0
        while bfs():
            for u in range(n):
                if pairU[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching == n

    best_m = 0
    best_L = 0

    # Scan all m values in one period
    for m in range(period):
        if time.time() - start_time > deadline:
            break

        # Binary search for minimum L
        lo, hi = 1, max_L
        min_L_for_m = max_L

        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m

        # Early termination if we hit the theoretical max
        if best_L >= max_L:
            break

    # Polish pass: check neighbors of best_m
    if time.time() - start_time < deadline - 3.0:
        for dm in range(-30, 31):
            if dm == 0:
                continue
            m2 = (best_m + dm) % period
            lo, hi = 1, max_L
            min_L_for_m2 = max_L
            while lo <= hi:
                mid = (lo + hi) // 2
                if has_perfect_matching(m2, mid):
                    min_L_for_m2 = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if min_L_for_m2 > best_L:
                best_L = min_L_for_m2
                best_m = m2

    if best_m < 0 or best_m > 1000000:
        best_m = 0

    return int(best_m)
