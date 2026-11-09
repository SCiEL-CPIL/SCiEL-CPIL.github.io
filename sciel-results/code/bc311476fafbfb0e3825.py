"""Fast bitmask greedy + local-search covering for C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0
    if t <= 1:
        return [list(range(min(k, v)))] if v <= k else [list(range(k))]
    if k >= v:
        return [list(range(v))]

    pts = range(v)
    tsubs = list(itertools.combinations(pts, t))
    n_t = len(tsubs)
    t_mask = []
    for c in tsubs:
        m = 0
        for x in c:
            m |= 1 << x
        t_mask.append(m)

    # all k-subsets as masks
    kmasks = []
    for c in itertools.combinations(pts, k):
        m = 0
        for x in c:
            m |= 1 << x
        kmasks.append(m)

    # covers[i] = list of block indices covering t-subset i
    covers = [[] for _ in range(n_t)]
    for b, bm in enumerate(kmasks):
        for i, tm in enumerate(t_mask):
            if (bm & tm) == tm:
                covers[i].append(b)

    # coverage mask per block (over t-subsets)
    bcov = [0] * len(kmasks)
    for b, bm in enumerate(kmasks):
        m = 0
        for i, tm in enumerate(t_mask):
            if (bm & tm) == tm:
                m |= 1 << i
        bcov[b] = m

    # ---- Greedy ----
    unc = (1 << n_t) - 1
    chosen = []
    while unc:
        if time.time() > deadline - 5:
            break
        i = (unc & -unc).bit_length() - 1
        best_b = None
        best_g = -1
        for b in covers[i]:
            g = (bcov[b] & unc).bit_count()
            if g > best_g:
                best_g = g
                best_b = b
        if best_b is None:
            break
        chosen.append(best_b)
        unc &= ~bcov[best_b]

    # convert to blocks (lists of points)
    blocks = []
    for b in chosen:
        bm = kmasks[b]
        blk = [x for x in range(v) if bm & (1 << x)]
        blocks.append(blk)

    # ---- Local search ----
    blocks = _local_search(blocks, v, k, t, n_t, t_mask, deadline)
    return blocks


def _local_search(blocks, v, k, t, n_t, t_mask, deadline):
    if not blocks:
        return blocks
    # coverage masks
    def cov_of(blk):
        bm = 0
        for x in blk:
            bm |= 1 << x
        m = 0
        for i, tm in enumerate(t_mask):
            if (bm & tm) == tm:
                m |= 1 << i
        return m

    covs = [cov_of(b) for b in blocks]

    # Phase 1: remove redundant blocks (coverage subset of union of others)
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        n = len(covs)
        for i in range(n):
            u = 0
            for j in range(n):
                if j != i:
                    u |= covs[j]
            if (covs[i] & ~u) == 0:
                del blocks[i]
                del covs[i]
                changed = True
                break

    # Phase 2: try to replace a block with a better one (heuristic: none cheap)
    return blocks
# EVOLVE-BLOCK-END
