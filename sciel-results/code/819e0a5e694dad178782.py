"""Rewritten: sliding-window + bipartite matching + binary search on width W."""
# EVOLVE-BLOCK-START
import math
import time

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = 10
_LCM = 270270  # lcm of the divisors


def _bipartite_match(avail):
    """Hopcroft-Karp. avail[i] = list of candidate value-indices for divisor i.
    Returns True if a perfect matching exists."""
    n = _N
    pair_u = [-1] * n
    pair_v = [-1] * (sum(len(a) for a in avail))
    # map (u, idx_in_avail[u]) -> global v index
    # we build global v index on the fly
    v_index = []
    for u in range(n):
        for v in avail[u]:
            v_index.append(v)
    # adjacency: for each u, list of global v ids
    adj = []
    gid = 0
    for u in range(n):
        lst = []
        for v in avail[u]:
            lst.append(gid)
            gid += 1
        adj.append(lst)

    from collections import deque

    def bfs():
        dist = [-1] * n
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
        found = False
        while q:
            u = q.popleft()
            for gv in adj[u]:
                u2 = pair_v[gv]
                if u2 == -1:
                    found = True
                elif dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
        return dist

    def dfs(u, dist):
        for gv in adj[u]:
            u2 = pair_v[gv]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2, dist)):
                pair_u[u] = gv
                pair_v[gv] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while True:
        dist = bfs()
        if dist is None:
            break
        progress = False
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u, dist):
                    matching += 1
                    progress = True
        if not progress:
            break
    return matching == n


def _find_optimal_window(deadline):
    """Search for minimal window width W with a feasible assignment."""
    divisors = _DIVISORS
    n = _N
    LCM = _LCM

    # Precompute for each divisor the residue-0 offset; multiples are k*div
    # For a window [L, L+W], divisor d has candidates k such that L <= k*d <= L+W
    # i.e. k in [ceil(L/d), floor((L+W)/d)]

    def feasible(L, W):
        """Check if window [L, L+W] admits a distinct multiple per divisor."""
        avail = []
        for d in divisors:
            lo = (L + d - 1) // d
            hi = (L + W) // d
            if hi < lo:
                return None
            # candidate values k*d
            # To keep matching small, cap candidates; but window small so fine
            vals = list(range(lo, hi + 1))
            # store the actual value k*d
            avail.append([v * d for v in vals])
        # distinctness: each chosen value must be unique across divisors
        # Hopcroft-Karp on value-identity: build bipartite divisors -> values
        # But values can be large; map value->index
        # For speed, only include values that appear in >=1 divisor list (all do)
        # Build value set
        val_set = set()
        for lst in avail:
            for x in lst:
                val_set.add(x)
        # If window is very wide, val_set huge; but we binary search small W first.
        # To bound, if len(val_set) > some cap, still ok for small W.
        # Build matching where right side = distinct values
        # For perf, if total candidates large, use a heuristic first.
        # Simple: use greedy then HK.
        # Map each value to global id
        val_to_id = {v: i for i, v in enumerate(val_set)}
        adj = []
        for lst in avail:
            adj.append([val_to_id[x] for x in lst])
        return _hk(adj, len(val_set))

    def _hk(adj, m):
        n2 = len(adj)
        pair_u = [-1] * n2
        pair_v = [-1] * m
        from collections import deque

        def bfs():
            dist = [-1] * n2
            q = deque()
            for u in range(n2):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
            found = False
            while q:
                u = q.popleft()
                for gv in adj[u]:
                    u2 = pair_v[gv]
                    if u2 == -1:
                        found = True
                    elif dist[u2] == -1:
                        dist[u2] = dist[u] + 1
                        q.append(u2)
            return dist, found

        def dfs(u, dist):
            for gv in adj[u]:
                u2 = pair_v[gv]
                if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2, dist)):
                    pair_u[u] = gv
                    pair_v[gv] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while True:
            dist, found = bfs()
            if not found:
                break
            progress = False
            for u in range(n2):
                if pair_u[u] == -1:
                    if dfs(u, dist):
                        matching += 1
                        progress = True
            if not progress:
                break
        return matching == n2

    # Lower bound on W: at least 1 (distinct ints). Also pigeonhole-ish.
    # We'll binary search W. Upper bound: try up to some cap.
    # Periodicity: feasibility depends on L mod LCM. So scan L in [0, LCM-1] for each W.
    # To keep within time, binary search W; for each W, scan L over a reduced set.
    # Since LCM=270270 is large, scanning all L for each W is too slow.
    # Instead: for a fixed W, feasibility as a function of L is periodic with period LCM.
    # But we can sample: the pattern of which multiples fall in window shifts with L.
    # Heuristic: try L = 0, then if fail, try random L, then structured.
    # Better: note that for small W, the set of feasible L is sparse.
    # Alternative: iterate W from lower bound upward, and for each W do a smart L search.

    # Lower bound: we need 10 distinct values, each a multiple of its d, within [L,L+W].
    # Minimal possible spread: sort divisors; but they must be distinct.
    # A simple LB: W >= 1. Let's start W at 1.

    # Upper bound heuristic: place each divisor's multiple at its d (i.e., k=1):
    # values = divisors themselves: max=39, min=6 -> W=33. But not distinct? 6,10,14,15,21,22,26,33,35,39 all distinct! 
    # So W=33 is feasible with L=6. Check: 6%6=0,10%10=0,...,39%39=0. All in [6,39]. Distinct. 
    # So answer <= 33. Can we do better?
    # Try to find W < 33.

    best_W = None
    best_L = None
    best_assign = None

    # We'll try W from a small lower bound up to 33 (known feasible), and for each W scan L smartly.
    # Since LCM is large, we can't scan all L. But for a given W, the window [L,L+W] contains
    # for divisor d the multiples k*d. The count is about W/d. For small W and large d, few.
    # Key: feasibility is periodic in L with period LCM. But we can reduce: 
    # For the assignment to exist, there must be distinct values. 
    # Let's try L in a range: since values are k*d, and we want them clustered, 
    # try L near multiples of small divisors.
    # Practical approach: for each W, try L = 0..LCM-1 is too much. 
    # Instead, note that if a solution exists at L, it exists at L+LCM. The *set* of values shifts by LCM.
    # For minimization of W, we just need existence. 
    # Heuristic: try L in [0, 2000] and also L near each divisor's multiples.
    # Given time budget, let's do: for W in range(1, 34): for L in candidate_Ls(W): check.
    # candidate_Ls: 0, and for each d, floor(x/d)*d for x in some range... 
    # Simpler: since we know W=33 works at L=6, and we want to find smaller W,
    # try W from 1 to 32, and for each, try L in a set of ~2000 smart points.

    # Smart L candidates: for a window to contain a multiple of d, L mod d must be such that
    # there's a multiple in [L, L+W]. For small W, this constrains L mod d.
    # Let's just try L = 0,1,...,2000 and also L = (k*d - W//2) for small k,d.
    # To be safe and fast, try L in [0, 3000] step 1 for small W (fast since matching small),
    # and if W is larger, reduce.

    # Actually for W up to 33, matching is tiny (each divisor has <= W/d+1 <= 6 candidates for d=6).
    # So checking one L is very fast. We can afford L in [0, 5000] for each W.

    for W in range(1, 34):
        if time.time() > deadline - 2:
            break
        found = False
        # Try L from 0 upward, but stop early. Also try L mod LCM coverage via small range.
        # For small W, a solution L if it exists likely < LCM, but we sample [0, 5000].
        for L in range(0, 5001):
            if time.time() > deadline - 2:
                break
            res = feasible(L, W)
            if res:
                # extract assignment
                # re-run to get pair_u
                divisors2 = _DIVISORS
                avail = []
                for d in divisors2:
                    lo = (L + d - 1) // d
                    hi = (L + W) // d
                    avail.append([v * d for v in range(lo, hi + 1)])
                val_set = set()
                for lst in avail:
                    for x in lst:
                        val_set.add(x)
                val_to_id = {v: i for i, v in enumerate(val_set)}
                adj = [[val_to_id[x] for x in lst] for lst in avail]
                # run HK to get assignment
                n2 = len(adj)
                m2 = len(val_set)
                pair_u = [-1] * n2
                pair_v = [-1] * m2
                from collections import deque

                def bfs2():
                    dist = [-1] * n2
                    q = deque()
                    for u in range(n2):
                        if pair_u[u] == -1:
                            dist[u] = 0
                            q.append(u)
                    found2 = False
                    while q:
                        u = q.popleft()
                        for gv in adj[u]:
                            u2 = pair_v[gv]
                            if u2 == -1:
                                found2 = True
                            elif dist[u2] == -1:
                                dist[u2] = dist[u] + 1
                                q.append(u2)
                    return dist, found2

                def dfs2(u, dist):
                    for gv in adj[u]:
                        u2 = pair_v[gv]
                        if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs2(u2, dist)):
                            pair_u[u] = gv
                            pair_v[gv] = u
                            return True
                    dist[u] = -1
                    return False

                matching = 0
                while True:
                    dist, found2 = bfs2()
                    if not found2:
                        break
                    progress = False
                    for u in range(n2):
                        if pair_u[u] == -1:
                            if dfs2(u, dist):
                                matching += 1
                                progress = True
                    if not progress:
                        break
                if matching == n2:
                    assign = [list(val_to_id.keys())[pair_u[u]] for u in range(n2)]
                    # verify distinct and valid
                    if len(set(assign)) == n and all(assign[i] % divisors[i] == 0 for i in range(n)) \
                       and all(L <= assign[i] <= L + W for i in range(n)):
                        best_W = W
                        best_L = L
                        best_assign = assign
                        found = True
                        break
        if found:
            break
        # if not found in [0,5000], try a few more L around known good region
        if not found:
            for L in list(range(0, 5001)) + [6, 39, 40, 41, 50, 100, 1000, 5000, 10000]:
                if L < 0 or L > 100000:
                    continue
                if time.time() > deadline - 2:
                    break
                res = feasible(L, W)
                if res:
                    divisors2 = _DIVISORS
                    avail = []
                    for d in divisors2:
                        lo = (L + d - 1) // d
                        hi = (L + W) // d
                        avail.append([v * d for v in range(lo, hi + 1)])
                    val_set = set()
                    for lst in avail:
                        for x in lst:
                            val_set.add(x)
                    val_to_id = {v: i for i, v in enumerate(val_set)}
                    adj = [[val_to_id[x] for x in lst] for lst in avail]
                    n2 = len(adj)
                    m2 = len(val_set)
                    pair_u = [-1] * n2
                    pair_v = [-1] * m2
                    from collections import deque

                    def bfs3():
                        dist = [-1] * n2
                        q = deque()
                        for u in range(n2):
                            if pair_u[u] == -1:
                                dist[u] = 0
                                q.append(u)
                        found3 = False
                        while q:
                            u = q.popleft()
                            for gv in adj[u]:
                                u2 = pair_v[gv]
                                if u2 == -1:
                                    found3 = True
                                elif dist[u2] == -1:
                                    dist[u2] = dist[u] + 1
                                    q.append(u2)
                        return dist, found3

                    def dfs3(u, dist):
                        for gv in adj[u]:
                            u2 = pair_v[gv]
                            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs3(u2, dist)):
                                pair_u[u] = gv
                                pair_v[gv] = u
                                return True
                        dist[u] = -1
                        return False

                    matching = 0
                    while True:
                        dist, found3 = bfs3()
                        if not found3:
                            break
                        progress = False
                        for u in range(n2):
                            if pair_u[u] == -1:
                                if dfs3(u, dist):
                                    matching += 1
                                    progress = True
                        if not progress:
                            break
                    if matching == n2:
                        assign = [list(val_to_id.keys())[pair_u[u]] for u in range(n2)]
                        if len(set(assign)) == n and all(assign[i] % divisors[i] == 0 for i in range(n)) \
                           and all(L <= assign[i] <= L + W for i in range(n)):
                            best_W = W
                            best_L = L
                            best_assign = assign
                            found = True
                            break
        if found:
            break

    if best_assign is not None:
        return best_assign

    # Fallback: the known feasible assignment (divisors themselves)
    return [int(d) for d in _DIVISORS]


def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    try:
        result = _find_optimal_window(deadline)
    except Exception:
        result = [int(d) for d in _DIVISORS]
    # Ensure types are plain int and list
    out = []
    for x in result:
        out.append(int(x))
    return out
# EVOLVE-BLOCK-END
