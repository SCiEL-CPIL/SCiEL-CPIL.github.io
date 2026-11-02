"""Vectorized covering design C(19,7,4): fast greedy build + local-search refinement."""
import time
import itertools
import numpy as np

T0 = time.time()
LIMIT = 150.0
MARGIN = 15.0


def _comb_indices(v, k):
    """All k-combinations of range(v) as a (C, k) int array."""
    return np.array(list(itertools.combinations(range(v), k)), dtype=np.int32)


def _covered_mask(covered, blocks, t):
    """Boolean array over C(v,t) t-subsets: covered if contained in any block."""
    n = len(covered)
    out = np.zeros(n, dtype=bool)
    for blk in blocks:
        b = np.array(blk, dtype=np.int32)
        sub = np.array(list(itertools.combinations(b, t)), dtype=np.int32)
        out[covered[sub].ravel()] = True
    return out


def _block_score(blk, uncovered, covered, t, v):
    """Number of currently-uncovered t-subsets contained in blk."""
    b = np.array(blk, dtype=np.int32)
    sub = np.array(list(itertools.combinations(b, t)), dtype=np.int32)
    idx = covered[sub].ravel()
    return int(np.count_nonzero(uncovered[idx]))


def _greedy_build(v, k, t, covered, deadline):
    """Vectorized greedy: repeatedly pick the k-subset covering most uncovered t-sets."""
    n = len(covered)
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    # Precompute point -> list of t-subset indices containing that point
    point_subs = []
    for x in range(v):
        lst = []
        for i, cs in enumerate(covered):
            if x in cs:
                lst.append(i)
        point_subs.append(np.array(lst, dtype=np.int32))

    while uncovered.any():
        if time.time() > deadline:
            break
        # Score each candidate point by uncovered t-sets it touches
        scores = np.array([uncovered[ps].sum() for ps in point_subs], dtype=np.int64)
        # Greedily build a block: pick point with max marginal gain
        block = []
        in_block = [False] * v
        # First point: highest score
        p = int(np.argmax(scores))
        block.append(p)
        in_block[p] = True
        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_p = -1
            for x in range(v):
                if in_block[x]:
                    continue
                # gain = uncovered t-sets that contain x AND all current block points
                mask = uncovered[point_subs[x]]
                for q in block:
                    mask = mask & (covered[point_subs[q]].ravel() >= 0)  # placeholder
                # Better: intersect via actual subset membership
                idx_x = point_subs[x]
                ok = np.ones(len(idx_x), dtype=bool)
                for q in block:
                    ok &= (covered[idx_x] == covered[point_subs[q]]).all(axis=1)
                gain = int((uncovered[idx_x] & ok).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_p = x
            if best_p < 0:
                break
            block.append(best_p)
            in_block[best_p] = True
        block.sort()
        blocks.append(block)
        # mark covered
        b = np.array(block, dtype=np.int32)
        sub = np.array(list(itertools.combinations(b, t)), dtype=np.int32)
        uncovered[covered[sub].ravel()] = False
    return blocks


def construct(v, k, t):
    deadline = T0 + LIMIT - MARGIN
    covered = _comb_indices(v, t)          # (C(v,t), t)
    n = len(covered)

    # ---- Phase 1: fast initial covering via randomized greedy ----
    best_blocks = None
    best_len = 10**9
    rng = np.random.default_rng(12345)

    # Simple fast greedy: repeatedly pick a random-ish k-set biased toward uncovered
    # We use a lighter heuristic: build blocks to cover the rarest uncovered t-sets.
    for trial in range(6):
        if time.time() > deadline:
            break
        uncovered = np.ones(n, dtype=bool)
        blocks = []
        # precompute for speed: for each t-subset index, its points
        while uncovered.any():
            if time.time() > deadline:
                break
            # pick an uncovered t-subset (the first / a random one)
            idxs = np.nonzero(uncovered)[0]
            pick = int(rng.choice(idxs))
            seed_pts = list(covered[pick])
            # complete to k points choosing points that cover most remaining uncovered
            block = seed_pts[:]
            inb = set(block)
            while len(block) < k:
                gains = []
                for x in range(v):
                    if x in inb:
                        continue
                    # count uncovered t-sets containing x and all of block
                    cnt = 0
                    for i in idxs:
                        cs = covered[i]
                        if x in cs and all(p in cs for p in block):
                            cnt += 1
                    gains.append((cnt, x))
                if not gains:
                    break
                gains.sort(reverse=True)
                # pick top with slight randomization
                top = gains[:max(1, len(gains) // 4)]
                x = int(rng.choice([g[1] for g in top]))
                block.append(x)
                inb.add(x)
            block.sort()
            blocks.append(block)
            b = np.array(block, dtype=np.int32)
            sub = np.array(list(itertools.combinations(b, t)), dtype=np.int32)
            uncovered[covered[sub].ravel()] = False
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]

    # If greedy never finished (deadline), build a guaranteed-valid covering as fallback
    if best_blocks is None or not np.all(_covered_mask(covered, best_blocks, t)):
        # deterministic exhaustive-safe fallback: include a block for each t-subset's points
        # (may exceed C(v,t) but guarantees validity; only used if we ran out of time)
        fb = []
        for i in range(n):
            blk = list(covered[i])
            while len(blk) < k:
                blk.append(len(blk))  # pad with new points if needed (v>=k assumed)
            blk = sorted(blk[:k])
            if blk not in fb:
                fb.append(blk)
        best_blocks = fb
        best_len = len(fb)

    # ---- Phase 2: local search to reduce block count ----
    if time.time() < deadline:
        blocks = best_blocks[:]
        covered_arr = _covered_mask(covered, blocks, t)
        # count coverage per t-subset
        cov_count = covered_arr.astype(np.int32)
        uncovered = cov_count == 0

        def try_remove(blocks, covered_arr, cov_count):
            """Remove any block that is redundant (all its t-sets still covered by others)."""
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if time.time() > deadline:
                        return blocks, covered_arr, cov_count, False
                    b = np.array(blocks[i], dtype=np.int32)
                    sub = np.array(list(itertools.combinations(b, t)), dtype=np.int32)
                    idx = covered[sub].ravel()
                    if (cov_count[idx] > 1).all():
                        cov_count[idx] -= 1
                        del blocks[i]
                        changed = True
            return blocks, covered_arr, cov_count, changed

        def try_swap(blocks, covered_arr, cov_count, deadline):
            """Try replacing one block with a new one that covers more uncovered t-sets."""
            if time.time() > deadline:
                return blocks, covered_arr, cov_count, False
            uncovered_idx = np.nonzero(cov_count == 0)[0]
            if len(uncovered_idx) == 0:
                return blocks, covered_arr, cov_count, False
            # For each existing block, try to find a better replacement
            improved = False
            # Build candidate replacement: pick a k-set maximizing uncovered coverage
            # Heuristic: use points from uncovered t-sets
            # Sample some uncovered t-sets, build a block around them, see if it helps net
            for _ in range(30):
                if time.time() > deadline:
                    break
                # choose a random existing block to potentially drop
                i = int(rng.integers(len(blocks)))
                b_old = np.array(blocks[i], dtype=np.int32)
                old_sub = np.array(list(itertools.combinations(b_old, t)), dtype=np.int32)
                old_idx = covered[old_sub].ravel()
                # only consider dropping if no t-set becomes uncovered
                if (cov_count[old_idx] <= 1).any():
                    continue
                # build a new block covering uncovered t-sets
                unc = uncovered_idx
                if len(unc) == 0:
                    break
                seed = list(covered[int(rng.choice(unc))])
                newblk = seed[:]
                inb = set(newblk)
                while len(newblk) < k:
                    best_g = -1
                    best_p = -1
                    for x in range(v):
                        if x in inb:
                            continue
                        cnt = 0
                        for j in unc:
                            cs = covered[j]
                            if x in cs and all(p in cs for p in newblk):
                                cnt += 1
                        if cnt > best_g:
                            best_g = cnt
                            best_p = x
                    if best_p < 0:
                        break
                    newblk.append(best_p)
                    inb.add(best_p)
                newblk.sort()
                if newblk == sorted(blocks[i]):
                    continue
                nb = np.array(newblk, dtype=np.int32)
                new_sub = np.array(list(itertools.combinations(nb, t)), dtype=np.int32)
                new_idx = covered[new_sub].ravel()
                # net uncovered change: new covers some uncovered; dropping old may uncover some
                gain = int(np.count_nonzero((cov_count[new_idx] == 0)))
                loss = int(np.count_nonzero((cov_count[old_idx] == 1)))
                if gain > loss:
                    cov_count[old_idx] -= 1
                    cov_count[new_idx] += 1
                    blocks[i] = list(newblk)
                    improved = True
            return blocks, covered_arr, cov_count, improved

        # Iterate removals and swaps
        for it in range(200):
            if time.time() > deadline:
                break
            blocks, covered_arr, cov_count, _ = try_remove(blocks, covered_arr, cov_count)
            if time.time() > deadline:
                break
            blocks, covered_arr, cov_count, sw = try_swap(blocks, covered_arr, cov_count, deadline)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(b) for b in blocks]
            if not sw and not uncovered.any():
                # no swaps helped; keep removing if possible
                blocks, covered_arr, cov_count, ch = try_remove(blocks, covered_arr, cov_count)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                if not ch:
                    break

    # Final validity safety: if somehow invalid, extend with one block per uncovered t-set
    final = [list(b) for b in best_blocks]
    cm = _covered_mask(covered, final, t)
    if not cm.all():
        missing = np.nonzero(~cm)[0]
        for i in missing:
            blk = list(covered[i])
            while len(blk) < k:
                blk.append(blk[-1] + 1 if len(blk) < v else 0)
            blk = sorted(set(blk))
            # ensure distinct and within range
            if len(blk) < k:
                for x in range(v):
                    if x not in blk:
                        blk.append(x)
                    if len(blk) == k:
                        break
            blk = sorted(blk[:k])
            if blk not in final:
                final.append(blk)
    return final
