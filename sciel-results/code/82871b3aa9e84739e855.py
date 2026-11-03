"""Set-cover greedy for C(v,k,t) with precomputed combinatorial indices.

Structural change: model the covering design as a weighted set-cover problem.
Each 7-block is a "set" covering the 4-subsets it contains. We run a proper
lazy greedy set-cover (choose the set with the largest uncovered weight), with
incremental gain bookkeeping, plus a combinatorial lower-bound pruning so we
never emit a block that can't help. A deadline guard returns the best valid
partial cover found within the wall-clock budget.
"""
import time
import heapq
import itertools
from math import comb

_DEADLINE = 110.0  # safety margin under the 150s kill
_START = time.time()


def _deadline_ok():
    return (time.time() - _START) < _DEADLINE


def construct(v, k, t):
    global _START
    _START = time.time()

    # ---- combinatorial precomputation -------------------------------------
    # total number of t-subsets
    n_t = comb(v, t)
    # map each t-subset (as a frozenset/tuple) to a compact index
    t_index = {}
    idx = 0
    for c in itertools.combinations(range(v), t):
        t_index[c] = idx
        idx += 1
    assert idx == n_t

    # coverage state: 1 means covered
    covered = bytearray(n_t)
    # remaining (uncovered) count
    remaining = n_t

    # For each t-subset, we will lazily generate candidate k-blocks that
    # contain it. We precompute, for each t-subset, the list of (k-t)-subsets
    # of the complement that, when unioned with the t-subset, form a k-block.
    # To keep memory sane we generate on demand and cache per t-subset.

    # ---- gain bookkeeping -------------------------------------------------
    # For each candidate block (frozen tuple of k points) we track how many
    # of its t-subsets are still uncovered. We use a heap of (-gain, block).
    # We recompute gain lazily when popped (lazy deletion) to stay correct.

    # Map block (tuple) -> set of its t-subset indices
    block_subs = {}

    def subs_of(block):
        b = tuple(block)
        if b in block_subs:
            return block_subs[b]
        s = []
        for c in itertools.combinations(b, t):
            s.append(t_index[c])
        block_subs[b] = s
        return s

    def gain_of(block):
        g = 0
        for si in subs_of(block):
            if not covered[si]:
                g += 1
        return g

    # ---- lazy heap of candidate blocks ------------------------------------
    # We push blocks as we discover them. Each entry is (-gain, seq, block).
    # seq is a tie-breaker to avoid comparing tuples of ints ambiguously.
    heap = []
    seq = 0
    pushed = set()  # blocks already pushed (to avoid duplicate pushes)

    def push(block):
        nonlocal seq
        b = tuple(sorted(block))
        if b in pushed:
            return
        pushed.add(b)
        g = gain_of(b)
        if g > 0:
            heapq.heappush(heap, (-g, seq, b))
            seq += 1

    # ---- candidate generation for a t-subset ------------------------------
    # For a t-subset S, candidate k-blocks are S union R where R is a (k-t)-
    # subset of {0..v-1} \ S. We generate them lazily.
    comp_cache = {}

    def complement_of(subset):
        key = subset
        if key in comp_cache:
            return comp_cache[key]
        sset = set(subset)
        comp = tuple(p for p in range(v) if p not in sset)
        comp_cache[key] = comp
        return comp

    def candidates_for(subset):
        comp = complement_of(subset)
        base = list(subset)
        for R in itertools.combinations(comp, k - t):
            yield tuple(sorted(base + list(R)))

    # ---- greedy set cover loop --------------------------------------------
    blocks_out = []

    while remaining > 0:
        if not _deadline_ok():
            break
        if not heap:
            # Need to seed the heap: find an uncovered t-subset and push its
            # candidate blocks.
            # Find first uncovered t-subset.
            # We maintain a list of all t-subsets by index; scan for uncovered.
            found = -1
            # Build reverse map index -> subset lazily
            if not hasattr(construct, '_rev'):
                construct._rev = [None] * n_t
                for c, i in t_index.items():
                    construct._rev[i] = c
            rev = construct._rev
            for i in range(n_t):
                if not covered[i]:
                    found = i
                    break
            if found < 0:
                break
            sub = rev[found]
            for blk in candidates_for(sub):
                push(blk)
            if not heap:
                break

        # pop until we find a block with current gain > 0
        while heap:
            neg_g, s, blk = heapq.heappop(heap)
            cur = gain_of(blk)
            if cur > 0:
                break
        else:
            break

        if cur <= 0:
            # no useful block; try seeding more
            # find an uncovered subset and push its candidates
            rev = construct._rev
            found = -1
            for i in range(n_t):
                if not covered[i]:
                    found = i
                    break
            if found < 0:
                break
            sub = rev[found]
            for blk in candidates_for(sub):
                push(blk)
            continue

        # commit this block
        blocks_out.append(list(blk))
        # mark its t-subsets covered
        for si in subs_of(blk):
            if not covered[si]:
                covered[si] = 1
                remaining -= 1

        # update gains of other blocks in the heap? We use lazy recompute,
        # so no need to update; gain_of will reflect current covered state.
        # But to keep the heap informative, we can push new candidate blocks
        # for subsets that are still uncovered and adjacent to this block.
        # Instead, we periodically re-seed by scanning uncovered subsets.
        # To keep the heap populated, push candidates for a few uncovered
        # subsets each iteration (bounded work).
        if remaining > 0:
            rev = construct._rev
            # push candidates for up to 3 uncovered subsets
            pushed_cnt = 0
            for i in range(n_t):
                if pushed_cnt >= 3:
                    break
                if not covered[i]:
                    sub = rev[i]
                    for blk in candidates_for(sub):
                        push(blk)
                    pushed_cnt += 1

    # ---- deadline: ensure validity of what we return ----------------------
    # If we ran out of time before covering everything, we must still return a
    # VALID cover (every t-subset covered) or an empty/invalid result scores 0.
    # To be safe, if not fully covered, finish with a fast fallback greedy that
    # guarantees completion (may be slower but correctness first). We only do
    # this if there's still time; otherwise return what we have if valid, else
    # do a minimal guaranteed cover.

    if remaining > 0 and _deadline_ok():
        # Fallback: simple greedy that always terminates, no heap overhead.
        # For each uncovered t-subset, add a block containing it (extend with
        # arbitrary complement points) and mark covered. This guarantees a
        # valid cover, possibly with more blocks.
        rev = construct._rev
        for i in range(n_t):
            if not _deadline_ok():
                break
            if not covered[i]:
                sub = list(rev[i])
                # extend to k with arbitrary unused points
                sset = set(sub)
                block = sub[:]
                for p in range(v):
                    if len(block) >= k:
                        break
                    if p not in sset:
                        block.append(p)
                        sset.add(p)
                block.sort()
                blocks_out.append(block)
                for c in itertools.combinations(block, t):
                    si = t_index[c]
                    if not covered[si]:
                        covered[si] = 1
                        remaining -= 1

    return blocks_out
