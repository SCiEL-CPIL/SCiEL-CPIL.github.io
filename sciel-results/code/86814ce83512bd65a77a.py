import time

def construct():
    deadline = time.time() + 80  # safety margin
    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144
    min_L = 12

    best_m = 12
    best_L = -1

    # Precompute for speed: for each i in 1..12, we'll use modulo.
    # For each m, build candidate values and their divisor bitmasks.

    for m in range(period):
        if time.time() > deadline:
            break
        # Build candidates m+1 .. m+max_L, with divisor bitmask for each
        # divmask[j] for candidate j (value = m+1+j), j in 0..max_L-1
        # bit i (0-indexed, i=0 -> index 1) set if (m+1+j) % (i+1) == 0
        divmask = [0]*max_L
        for j in range(max_L):
            v = m + 1 + j
            mask = 0
            for i in range(1, n+1):
                if v % i == 0:
                    mask |= (1 << (i-1))
            divmask[j] = mask

        # For a given L, adjacency: for each left index i (0..11), list of j in 0..L-1 with bit i set
        def feasible(L):
            # build adj for left nodes
            adj = [[] for _ in range(n)]
            for j in range(L):
                msk = divmask[j]
                i = 0
                while msk:
                    if msk & 1:
                        adj[i].append(j)
                    msk >>= 1
                    i += 1
            # Kuhn
            matchR = [-1]*L
            # order left nodes by degree ascending for early failure
            order = sorted(range(n), key=lambda i: len(adj[i]))
            for i in order:
                if not adj[i]:
                    return False
                seen = [False]*L
                # iterative or recursive dfs
                def dfs(x):
                    for j in adj[x]:
                        if not seen[j]:
                            seen[j] = True
                            if matchR[j] == -1 or dfs(matchR[j]):
                                matchR[j] = x
                                return True
                    return False
                if not dfs(i):
                    return False
            return True

        # binary search max feasible L in [min_L, max_L]
        lo, hi = min_L, max_L
        ans = min_L - 1
        # check if even min_L feasible
        # binary search for largest feasible
        while lo <= hi:
            mid = (lo + hi)//2
            if feasible(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        L = ans
        if L > best_L:
            best_L = L
            best_m = m

    return int(best_m)
