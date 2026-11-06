"""Independent set in the square-sum conflict graph on [1,256]."""
import math
import time
import random

_DEADLINE = 75.0  # safety margin under the 90s kill
_START = time.monotonic()

def _is_square(n):
    if n <= 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_graph():
    n = 256
    # adjacency bitsets
    adj = [0] * (n + 1)  # 1-indexed
    for a in range(1, n + 1):
        bits = 0
        for b in range(1, n + 1):
            if a + b <= 512 and _is_square(a + b):
                bits |= 1 << b
        adj[a] = bits
    # diagonal: a+a square -> vertex is self-conflict (cannot be selected)
    self_conf = [0] * (n + 1)
    for a in range(1, n + 1):
        if _is_square(2 * a):
            self_conf[a] = 1
    return adj, self_conf

_ADJ, _SELF = _build_graph()
_N = 256
# vertices that are not self-conflicting (eligible)
_ELIG = [a for a in range(1, _N + 1) if _SELF[a] == 0]
_ELIG_SET = 0
for a in _ELIG:
    _ELIG_SET |= 1 << a

# reorder vertices by degree (ascending) for better B&B branching; keep a map
_deg = [bin(_ADJ[a] & _ELIG_SET).count("1") for a in range(_N + 1)]
_order = sorted(_ELIG, key=lambda a: _deg[a])
# position map: vertex -> index in _order
_pos = {a: i for i, a in enumerate(_order)}
_m = len(_order)
# permuted adjacency in the new order
_perm_adj = [0] * _m
for i, a in enumerate(_order):
    bits = _ADJ[a] & _ELIG_SET
    pb = 0
    for j, b in enumerate(_order):
        if bits & (1 << b):
            pb |= 1 << j
    _perm_adj[i] = pb

def _max_independent_bound(adj, m, avail_mask):
    # greedy coloring upper bound on independent set size in avail_mask
    # returns (color_count, color_of_vertex) for vertices in avail
    # We color the subgraph induced by avail.
    uncolored = avail_mask
    colors = {}
    color_classes = []  # list of masks
    while uncolored:
        # pick a vertex (lowest index)
        v = (uncolored & -uncolored).bit_length() - 1
        uncolored &= uncolored - 1
        # assign smallest color not used by colored neighbors
        used = set()
        nb = adj[v] & avail_mask
        # neighbors already colored
        for c_idx, cmask in enumerate(color_classes):
            if nb & cmask:
                used.add(c_idx)
        c = 0
        while c in used:
            c += 1
        if c == len(color_classes):
            color_classes.append(0)
        color_classes[c] |= 1 << v
        colors[v] = c
    return len(color_classes)

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

def _local_search(seed_mask, deadline):
    # hill-climb: try adding a vertex, if conflicts, try swaps
    cur = seed_mask
    cur_size = cur.bit_count()
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # try add each eligible vertex not in cur
        for a in _ELIG:
            bit = 1 << a
            if cur & bit:
                continue
            if _ADJ[a] & cur:
                continue
            cur |= bit
            cur_size += 1
            improved = True
        # try swaps: remove one, add another to gain
        for a in _ELIG:
            bit = 1 << a
            if cur & bit:
                continue
            if not (_ADJ[a] & cur):
                continue
            # conflicts with cur
            conflicts = _ADJ[a] & cur
            # try removing one conflict vertex c to add a
            tmp_conf = conflicts
            while tmp_conf:
                c = (tmp_conf & -tmp_conf).bit_length() - 1
                tmp_conf &= tmp_conf - 1
                # check a doesn't conflict with cur minus c
                if _ADJ[a] & (cur & ~(1 << c)) == 0:
                    # also c removal must not lose more than gain (net +0 or +1)
                    # we replace c with a: size same, but maybe enables others
                    new_cur = (cur & ~(1 << c)) | bit
                    # only accept if it opens more: do a quick add pass
                    added = new_cur
                    for b in _ELIG:
                        bb = 1 << b
                        if added & bb:
                            continue
                        if _ADJ[b] & added:
                            continue
                        added |= bb
                    if added.bit_count() > cur_size:
                        cur = added
                        cur_size = added.bit_count()
                        improved = True
                        break
    return cur

def construct():
    start = time.monotonic()
    deadline = start + _DEADLINE
    random.seed(12345)

    best_mask = 0
    best_size = 0

    # Greedy initial: min-degree greedy
    def greedy_min_degree(order):
        chosen = 0
        avail = _ELIG_SET
        for a in order:
            if avail & (1 << a):
                if _ADJ[a] & chosen == 0:
                    chosen |= 1 << a
        return chosen

    # several randomized orderings
    for it in range(200):
        if time.monotonic() > deadline - 5:
            break
        # random permutation of eligible
        perm = list(_ELIG)
        random.shuffle(perm)
        mask = greedy_min_degree(perm)
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask

    # local search on best
    if time.monotonic() < deadline - 10:
        ls = _local_search(best_mask, deadline - 5)
        if ls.bit_count() > best_size:
            best_size = ls.bit_count()
            best_mask = ls

    # branch and bound on the whole graph if time permits (use a smaller deadline)
    if time.monotonic() < deadline - 30:
        bnd = _branch_and_bound(_perm_adj, _m, (1 << _m) - 1, 0, deadline - 5)
        # convert perm mask back to original
        orig = 0
        for i, a in enumerate(_order):
            if bnd & (1 << i):
                orig |= 1 << a
        if orig.bit_count() > best_size:
            best_size = orig.bit_count()
            best_mask = orig

    # final local search
    if time.monotonic() < deadline - 2:
        ls = _local_search(best_mask, deadline - 1)
        if ls.bit_count() > best_size:
            best_size = ls.bit_count()
            best_mask = ls

    result = [a for a in _ELIG if best_mask & (1 << a)]
    if not result:
        # fallback: any single eligible vertex
        result = [_ELIG[0]]
    return result

if __name__ == "__main__":
    print(construct())
