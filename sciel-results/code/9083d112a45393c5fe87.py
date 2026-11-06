<<<<<<< SEARCH
def _branch_and_bound(adj, m, avail_mask, current, deadline):
    # returns best independent set mask found
    best = [current]
    best_size = [current.bit_count()]
    # order vertices by index (already degree-ordered)
    # Use recursive branch and bound with coloring bound
    import sys
    sys.setrecursionlimit(10000)

    def rec(avail, cur, cur_size):
        if time.monotonic() > deadline:
            return
        if cur_size > best_size[0]:
            best_size[0] = cur_size
            best[0] = cur
        if avail == 0:
            return
        # bound
        ub = _max_independent_bound(adj, m, avail)
        if cur_size + ub <= best_size[0]:
            return
        # pick vertex: highest degree in avail for better pruning
        # find vertex with max degree within avail
        v = -1
        maxd = -1
        tmp = avail
        while tmp:
            x = (tmp & -tmp).bit_length() - 1
            tmp &= tmp - 1
            d = (adj[x] & avail).bit_count()
            if d > maxd:
                maxd = d
                v = x
        # branch 1: include v
        rec(avail & ~adj[v] & ~(1 << v), cur | (1 << v), cur_size + 1)
        # branch 2: exclude v
        rec(avail & ~(1 << v), cur, cur_size)

    rec(avail_mask, 0, 0)
    return best[0]
=======
def _branch_and_bound(adj, m, avail_mask, current, deadline):
    # Returns best independent set mask found
    best = [current]
    best_size = [current.bit_count()]
    import sys
    sys.setrecursionlimit(20000)

    def rec(avail, cur, cur_size):
        nonlocal best, best_size
        if time.monotonic() > deadline:
            return
        if cur_size > best_size[0]:
            best_size[0] = cur_size
            best[0] = cur
        if avail == 0:
            return
        # Quick bound: popcount of avail
        avail_count = avail.bit_count()
        if cur_size + avail_count <= best_size[0]:
            return
        # Greedy coloring bound
        ub = _max_independent_bound(adj, m, avail)
        if cur_size + ub <= best_size[0]:
            return
        # Pick vertex with max degree in avail
        v = -1
        maxd = -1
        tmp = avail
        while tmp:
            x = (tmp & -tmp).bit_length() - 1
            tmp &= tmp - 1
            d = (adj[x] & avail).bit_count()
            if d > maxd:
                maxd = d
                v = x
        if v == -1:
            return
        # Branch 1: include v
        rec(avail & ~adj[v] & ~(1 << v), cur | (1 << v), cur_size + 1)
        if time.monotonic() > deadline:
            return
        # Branch 2: exclude v
        rec(avail & ~(1 << v), cur, cur_size)

    rec(avail_mask, 0, 0)
    return best[0]
>>>>>>> REPLACE

