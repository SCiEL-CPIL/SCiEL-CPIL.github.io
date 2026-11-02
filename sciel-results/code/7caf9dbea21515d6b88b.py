"""Greedy bit-mask covering with local-search polish and a wall-clock guard."""
import time
import itertools


def _build_index(v, t):
    """Map each t-subset of 0..v-1 to a dense index 0..T-1 and back."""
    combos = list(itertools.combinations(range(v), t))
    T = len(combos)
    idx = {}
    for j, c in enumerate(combos):
        idx[c] = j
    return idx, combos, T


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s

    idx, combos, T = _build_index(v, t)
    nwords = (T + 63) // 64

    # cover[x]: list of nwords 64-bit ints; bit b set iff t-subset b contains x.
    cover = [[0] * nwords for _ in range(v)]
    for c, b in idx.items():
        w, bit = b >> 6, 1 << (b & 63)
        for x in c:
            cover[x][w] |= bit

    # uncovered: list of nwords ints; bit set iff that t-subset is not yet covered.
    uncovered = [(1 << 64) - 1 if i < nwords - 1 else ((1 << (T - (nwords - 1) * 64)) - 1)
                 for i in range(nwords)]
    # fix last word to only T bits
    last_bits = T - (nwords - 1) * 64
    uncovered[-1] = (1 << last_bits) - 1

    def popcount_word(w):
        return bin(w).count('1')

    def total_uncovered():
        return sum(popcount_word(w) for w in uncovered)

    def gains():
        g = [0] * v
        for x in range(v):
            cx = cover[x]
            s = 0
            for i in range(nwords):
                s += popcount_word(uncovered[i] & cx[i])
            g[x] = s
        return g

    blocks = []

    while total_uncovered() > 0:
        if time.time() > deadline - 25.0:
            break
        # find lowest set bit index b in uncovered
        b = None
        for w in range(nwords):
            if uncovered[w]:
                lb = uncovered[w] & (-uncovered[w])
                b = w * 64 + (lb.bit_length() - 1)
                break
        if b is None:
            break

        seed = list(combos[b])
        block = list(seed)
        in_block = [False] * v
        for x in block:
            in_block[x] = True

        g = gains()
        while len(block) < k:
            best_x, best_g = -1, -1
            for x in range(v):
                if in_block[x]:
                    continue
                if g[x] > best_g:
                    best_g, best_x = g[x], x
            if best_x < 0:
                break
            block.append(best_x)
            in_block[best_x] = True
            # update gains: subsets containing best_x AND all other block points are now
            # "in-block"; but they're only covered when the block is finalized. For the
            # marginal gain of adding more points we just recompute after each add is
            # expensive; instead recompute gains once per block (cheap enough).
            g = gains()

        block.sort()
        blocks.append(block)

        # mark all t-subsets of this block as covered
        for c in itertools.combinations(block, t):
            bb = idx[c]
            uncovered[bb >> 6] &= ~(1 << (bb & 63))

    # ---- validity / completion ----
    def covered_set(blks):
        s = set()
        for bl in blks:
            for c in itertools.combinations(bl, t):
                s.add(c)
        return s

    seen = covered_set(blocks)
    if len(seen) < T:
        for c in combos:
            if c not in seen:
                blk = list(c)
                for x in range(v):
                    if len(blk) == k:
                        break
                    if x not in blk:
                        blk.append(x)
                blk.sort()
                blocks.append(blk)
                seen.add(c)

    best = blocks

    # ---- polish: drop redundant blocks ----
    def count_map(blks):
        m = {}
        for bl in blks:
            for c in itertools.combinations(bl, t):
                m[c] = m.get(c, 0) + 1
        return m

    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        cm = count_map(best)
        for i in range(len(best)):
            if time.time() > deadline - 5.0:
                break
            redundant = all(cm[c] > 1 for c in itertools.combinations(best[i], t))
            if redundant:
                for c in itertools.combinations(best[i], t):
                    cm[c] -= 1
                best.pop(i)
                changed = True
                break

    # final safety: ensure validity
    seen = covered_set(best)
    if len(seen) < T:
        for c in combos:
            if c not in seen:
                blk = list(c)
                for x in range(v):
                    if len(blk) == k:
                        break
                    if x not in blk:
                        blk.append(x)
                blk.sort()
                best.append(blk)
                seen.add(c)

    return best
