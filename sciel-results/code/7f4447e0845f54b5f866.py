# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    start = time.time()
    deadline = start + 120.0
    margin = 6.0

    def remaining():
        return deadline - margin - time.time()

    # All t-subsets as bitmasks (t small, e.g. 3)
    all_ts = list(itertools.combinations(range(v), t))
    all_ts.sort()
    nT = len(all_ts)
    ts_mask = []
    for c in all_ts:
        m = 0
        for p in c:
            m |= (1 << p)
        ts_mask.append(m)
    ts_set = set(ts_mask)

    def block_to_mask(block):
        m = 0
        for p in block:
            m |= (1 << p)
        return m

    # Precompute: for each point p, the list of t-subset indices containing p
    point_to_ts = [[] for _ in range(v)]
    for i, m in enumerate(ts_mask):
        mm = m
        while mm:
            lb = mm & (-mm)
            p = lb.bit_length() - 1
            point_to_ts[p].append(i)
            mm -= lb

    # Precompute: for each t-subset index, the list of points in it (for quick iteration)
    ts_points = [tuple(c) for c in all_ts]

    # coverage: for each t-subset, number of blocks covering it (int list, dense)
    # We'll maintain coverage counts and uncovered set.

    def uncovered_count(coverage):
        # count zeros
        c = 0
        for x in coverage:
            if x == 0:
                c += 1
        return c

    def greedy_build():
        coverage = [0] * nT
        blocks = []
        # uncovered as a set of indices
        uncovered = set(range(nT))
        while uncovered:
            # start from a random uncovered subset (first for determinism)
            seed_idx = next(iter(uncovered))
            block = list(ts_points[seed_idx])
            # choose points to maximize coverage of uncovered
            while len(block) < k:
                best_p = -1
                best_gain = -1
                in_block = set(block)
                for p in range(v):
                    if p in in_block:
                        continue
                    # gain = number of uncovered t-subsets that would be covered
                    # A t-subset is covered by block if all its points are in block.
                    # Adding p covers those uncovered t-subsets that contain p and whose
                    # other t-1 points are already in block.
                    gain = 0
                    # iterate t-subsets containing p
                    # but only those where the other points are in block
                    for idx in point_to_ts[p]:
                        if coverage[idx] != 0:
                            continue
                        # check other points in ts are in block
                        ok = True
                        for q in ts_points[idx]:
                            if q == p:
                                continue
                            if q not in in_block:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1 or best_gain == 0:
                    # pick any not in block
                    for p in range(v):
                        if p not in in_block:
                            best_p = p
                            break
                block.append(best_p)
            block.sort()
            blocks.append(block)
            # update coverage for all t-subsets of this block
            bm = block_to_mask(block)
            for i, m in enumerate(ts_mask):
                if (m & bm) == m:
                    if coverage[i] == 0:
                        uncovered.discard(i)
                    coverage[i] += 1
        return blocks, coverage

    # Local search: try to remove a block if all its t-subsets are covered by others.
    def try_remove(blocks, coverage, blocks_of_ts):
        # blocks_of_ts: list of set of block indices covering each t-subset
        # We'll recompute a coverage structure that tracks counts.
        n = len(blocks)
        # For each block, list of t-subset indices it covers
        block_ts = []
        for b in blocks:
            bm = block_to_mask(b)
            lst = []
            for i, m in enumerate(ts_mask):
                if (m & bm) == m:
                    lst.append(i)
            block_ts.append(lst)
        # count per t-subset
        cov = [0] * nT
        for lst in block_ts:
            for i in lst:
                cov[i] += 1
        # try removing blocks in random order
        order = list(range(n))
        random.shuffle(order)
        removed = [False] * n
        for bi in order:
            if removed[bi]:
                continue
            # check if every t-subset in block_ts[bi] has cov >= 2
            ok = True
            for i in block_ts[bi]:
                if cov[i] < 2:
                    ok = False
                    break
            if ok:
                removed[bi] = True
                for i in block_ts[bi]:
                    cov[i] -= 1
        newblocks = [b for idx, b in enumerate(blocks) if not removed[idx]]
        return newblocks

    def local_search(blocks, budget):
        # repeat removal passes until no improvement or budget
        blocks = try_remove(blocks, None, None)
        # After removal, try to re-greedy-fill? No, removal only.
        # We can also try swapping: but keep it simple: removal-based reduction.
        return blocks

    def improve(blocks, budget):
        # Perform several rounds of removal-based local search.
        best = blocks
        while budget > 0 and time.time() < deadline - margin:
            cand = try_remove(best, None, None)
            if len(cand) < len(best):
                best = cand
            else:
                break
        return best

    # Strategy:
    # 1) Build an initial greedy covering.
    # 2) Improve via removal local search.
    # 3) If time remains, do additional greedy builds with different randomizations and keep best.

    best_blocks = None
    best_count = None

    # First greedy build (deterministic-ish)
    if remaining() <= 0:
        # extremely little time; do a minimal valid-ish attempt
        # fallback: just return a trivial set? Must be valid. Do a quick greedy.
        blocks, _ = greedy_build()
        return blocks

    blocks, _ = greedy_build()
    best_blocks = blocks
    best_count = len(blocks)

    # Improve
    if remaining() > 0:
        improved = improve(best_blocks, remaining())
        if len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)

    # Additional randomized greedy builds if time allows
    rng = random.Random(12345)
    while remaining() > 2.0:
        # randomized greedy: shuffle point order / seed choice
        # We'll do a quick greedy with random tie-breaking by reordering range(v)
        # For simplicity, run greedy_build with a random seed selection each time.
        # But greedy_build is deterministic; add randomness by shuffling the uncovered iteration.
        # We'll just run a fresh greedy (deterministic) -> same result. So instead,
        # try randomized local search: random removal order is already inside try_remove.
        # To get diversity, do another greedy but with a different starting point:
        # We'll modify by shuffling all_ts order for the greedy.
        # Implement a lightweight randomized greedy inline.
        coverage = [0] * nT
        uncovered = set(range(nT))
        blocks2 = []
        # random order of t-subsets to pick seeds
        order = list(range(nT))
        rng.shuffle(order)
        # We still need to process uncovered; use a random pick each time.
        while uncovered:
            seed_idx = rng.choice(list(uncovered))
            block = list(ts_points[seed_idx])
            while len(block) < k:
                best_p = -1
                best_gain = -1
                in_block = set(block)
                # randomize point order
                pts = list(range(v))
                rng.shuffle(pts)
                for p in pts:
                    if p in in_block:
                        continue
                    gain = 0
                    for idx in point_to_ts[p]:
                        if coverage[idx] != 0:
                            continue
                        ok = True
                        for q in ts_points[idx]:
                            if q == p:
                                continue
                            if q not in in_block:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1 or best_gain == 0:
                    for p in pts:
                        if p not in in_block:
                            best_p = p
                            break
                block.append(best_p)
            block.sort()
            blocks2.append(block)
            bm = block_to_mask(block)
            for i, m in enumerate(ts_mask):
                if (m & bm) == m:
                    if coverage[i] == 0:
                        uncovered.discard(i)
                    coverage[i] += 1
        # improve this candidate
        cand = improve(blocks2, remaining())
        if len(cand) < best_count:
            best_blocks = cand
            best_count = len(cand)

    return best_blocks
# EVOLVE-BLOCK-END
