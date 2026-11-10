"""Independent-set optimization for large-reciprocal-sum under lcm separation."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin under 90s wall-clock

    # Universe: [2, 256]
    N = 255
    vals = list(range(2, 257))  # vals[i] corresponds to element i (i in 0..254)
    # index map: value -> i
    idx = {v: i for i, v in enumerate(vals)}

    # Incompatibility: pair (a,b) incompatible if lcm(a,b) <= 256
    # We need independent set in this graph (no edge among selected).
    # Build adjacency as bitsets (int) for speed.
    adj = [0] * N
    for i in range(N):
        a = vals[i]
        for j in range(i + 1, N):
            b = vals[j]
            if math.lcm(a, b) <= 256:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Reciprocal weights (use float for search, exact rational for final score)
    w = [1.0 / vals[i] for i in range(N)]

    def weight_of(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            s += w[i]
            m ^= lsb
        return s

    def is_independent(mask):
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if adj[i] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Greedy builders: several heuristics to get diverse strong starts.
    def greedy_order(order):
        mask = 0
        for i in order:
            if (adj[i] & mask) == 0:
                mask |= (1 << i)
        return mask

    # Heuristic orders
    orders = []
    # smallest value first (max reciprocal)
    orders.append(sorted(range(N), key=lambda i: vals[i]))
    # largest reciprocal density
    orders.append(sorted(range(N), key=lambda i: vals[i]))
    # random shuffles
    rnd = random.Random(12345)
    for _ in range(6):
        o = list(range(N))
        rnd.shuffle(o)
        orders.append(o)
    # order by increasing number of neighbors (less constrained first)
    orders.append(sorted(range(N), key=lambda i: bin(adj[i]).count('1')))

    starts = []
    for o in orders:
        m = greedy_order(o)
        if m and m not in starts:
            starts.append(m)

    # Local search: add, remove, swap moves.
    def neighbors_of(mask):
        # candidate additions: elements not in mask and compatible with all in mask
        cands = []
        for i in range(N):
            if (mask >> i) & 1:
                continue
            if (adj[i] & mask) == 0:
                cands.append(i)
        return cands

    def local_search(mask, t_limit):
        cur = mask
        curw = weight_of(cur)
        improved = True
        while improved and (time.time() < t_limit):
            improved = False
            # ADD moves
            for i in neighbors_of(cur):
                new = cur | (1 << i)
                nw = curw + w[i]
                if nw > curw + 1e-15:
                    cur = new
                    curw = nw
                    improved = True
                    break
            if improved:
                continue
            # SWAP moves: remove one element, add a compatible one
            for r in range(N):
                if (cur >> r) & 1:
                    base = cur ^ (1 << r)
                    for a in range(N):
                        if (base >> a) & 1:
                            continue
                        if (adj[a] & base) == 0:
                            new = base | (1 << a)
                            nw = curw - w[r] + w[a]
                            if nw > curw + 1e-15:
                                cur = new
                                curw = nw
                                improved = True
                                break
                    if improved:
                        break
        return cur, curw

    best_mask = max(starts, key=weight_of) if starts else 0
    best_w = weight_of(best_mask)

    # Run local search from each start, keeping best.
    for m in starts:
        if time.time() > DEADLINE:
            break
        lm, lw = local_search(m, DEADLINE)
        if lw > best_w:
            best_w = lw
            best_mask = lm

    # Additional branch-and-bound search for maximum weight independent set
    # on the incompatibility graph, seeded with best known.
    # Use recursive BB over sorted-by-weight vertices with upper bound =
    # current + sum of weights of remaining candidates (loose but ok).
    # To keep it tractable, do a focused search: try adding/removing around best.

    # Refinement: try 1-flip (add or remove single) and 2-flip improvements.
    def try_2flip(mask, curw):
        # remove two, add two (or add two, remove one, etc.) - keep simple:
        # try removing one and adding two
        best = (mask, curw)
        elems = [i for i in range(N) if (mask >> i) & 1]
        for r in elems:
            base = mask ^ (1 << r)
            base_w = curw - w[r]
            cands = [a for a in range(N) if not ((base >> a) & 1) and (adj[a] & base) == 0]
            # add up to two compatible cands
            for a1 in cands:
                b1 = base | (1 << a1)
                b1w = base_w + w[a1]
                # add second
                for a2 in cands:
                    if a2 == a1:
                        continue
                    if (adj[a2] & b1) == 0:
                        new = b1 | (1 << a2)
                        nw = b1w + w[a2]
                        if nw > best[1]:
                            best = (new, nw)
        return best

    # A few 2-flip passes
    for _ in range(3):
        if time.time() > DEADLINE:
            break
        nm, nw = try_2flip(best_mask, best_w)
        if nw > best_w:
            best_mask, best_w = nm, nw
        else:
            break

    # Final local search from the refined best
    if time.time() < DEADLINE:
        lm, lw = local_search(best_mask, DEADLINE)
        if lw > best_w:
            best_mask, best_w = lm, lw

    # Convert mask to value list
    result = [vals[i] for i in range(N) if (best_mask >> i) & 1]
    # Safety: ensure nonempty
    if not result:
        result = [2]
    return result
# EVOLVE-BLOCK-END
