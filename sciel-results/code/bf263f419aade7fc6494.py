import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - t0

    # Build universe
    pts = list(range(v))
    all_triples = list(itertools.combinations(pts, t))
    all_blocks = list(itertools.combinations(pts, k))
    n_triples = len(all_triples)
    n_blocks = len(all_blocks)

    tri_idx = {tr: i for i, tr in enumerate(all_triples)}

    # For each block, precompute:
    #   - a bitmask (Python int) with bit j set iff triple j is covered
    #   - the list of triple indices it covers
    block_mask = [0] * n_blocks
    block_tris = [[] for _ in range(n_blocks)]
    for bi, blk in enumerate(all_blocks):
        m = 0
        lst = []
        for sub in itertools.combinations(blk, t):
            ti = tri_idx[sub]
            m |= (1 << ti)
            lst.append(ti)
        block_mask[bi] = m
        block_tris[bi] = lst

    # Reverse index: triple -> list of block indices covering it
    tri2blocks = [[] for _ in range(n_triples)]
    for bi in range(n_blocks):
        for ti in block_tris[bi]:
            tri2blocks[ti].append(bi)

    # Greedy set cover
    def run_greedy(seed_val):
        rng = random.Random(seed_val)
        # Randomize block order for tie-breaking diversity
        order = list(range(n_blocks))
        rng.shuffle(order)

        uncovered = (1 << n_triples) - 1
        chosen = []
        used = set()

        while uncovered:
            if elapsed() > DEADLINE - 12:
                break
            best_g = -1
            best_i = -1
            for bi in order:
                if bi in used:
                    continue
                g = (uncovered & block_mask[bi]).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = bi
            if best_i < 0 or best_g == 0:
                break
            chosen.append(best_i)
            used.add(best_i)
            uncovered &= ~block_mask[best_i]
        return chosen, used

    # Local search: iterative removal + 1-swap + 2-for-1
    def polish(blocks_in):
        if not blocks_in:
            return list(blocks_in)
        cur = list(blocks_in)
        cur_set = set(cur)

        # Coverage counts
        cnt = [0] * n_triples
        for bi in cur:
            for ti in block_tris[bi]:
                cnt[ti] += 1

        improved = True
        while improved and elapsed() < DEADLINE - 4:
            improved = False

            # --- Pass 1: remove redundant blocks ---
            keep = []
            for bi in cur:
                removable = True
                for ti in block_tris[bi]:
                    if cnt[ti] == 1:
                        removable = False
                        break
                if removable:
                    for ti in block_tris[bi]:
                        cnt[ti] -= 1
                    cur_set.discard(bi)
                    improved = True
                else:
                    keep.append(bi)
            cur = keep

            if not cur:
                break

            # --- Pass 2: 1-for-1 swaps ---
            for idx, bi in enumerate(cur):
                if elapsed() > DEADLINE - 4:
                    break
                critical = [ti for ti in block_tris[bi] if cnt[ti] == 1]
                if not critical:
                    # redundant, remove
                    for ti in block_tris[bi]:
                        cnt[ti] -= 1
                    cur_set.discard(bi)
                    cur[idx] = None
                    improved = True
                    continue

                # Find best candidate: covers all critical, minimizes new overlap
                first = critical[0]
                cands = tri2blocks[first]
                best_cand = -1
                best_new = 10**9
                for ck in cands:
                    if ck in cur_set:
                        continue
                    ok = True
                    for ti in critical:
                        if not (block_mask[ck] >> ti) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Count how many new triples this block adds
                    new_cnt = 0
                    for ti in block_tris[ck]:
                        if cnt[ti] == 0:
                            new_cnt += 1
                    if new_cnt < best_new:
                        best_new = new_cnt
                        best_cand = ck

                if best_cand >= 0:
                    # Remove bi
                    for ti in block_tris[bi]:
                        cnt[ti] -= 1
                    cur_set.discard(bi)
                    # Add best_cand
                    for ti in block_tris[best_cand]:
                        cnt[ti] += 1
                    cur_set.add(best_cand)
                    cur[idx] = best_cand
                    improved = True
                    break  # restart passes

            # --- Pass 3: 2-for-1 (limited) ---
            if len(cur) > 1 and elapsed() < DEADLINE - 8:
                n_cur = len(cur)
                rng2 = random.Random(int(elapsed() * 1000) & 0xFFFF)
                indices = list(range(n_cur))
                rng2.shuffle(indices)
                max_checks = 3000
                checks = 0
                found = False
                for a in range(len(indices)):
                    if checks >= max_checks or elapsed() > DEADLINE - 4:
                        break
                    for b in range(a + 1, len(indices)):
                        if checks >= max_checks or elapsed() > DEADLINE - 4:
                            break
                        checks += 1
                        bi = cur[indices[a]]
                        bj = cur[indices[b]]
                        if bi is None or bj is None:
                            continue
                        # Required: triples covered only by bi or bj
                        req = []
                        seen = set()
                        for ti in block_tris[bi]:
                            if cnt[ti] == 1 and ti not in seen:
                                req.append(ti)
                                seen.add(ti)
                        for ti in block_tris[bj]:
                            if cnt[ti] == 1 and ti not in seen:
                                req.append(ti)
                                seen.add(ti)
                        if not req:
                            continue
                        # Most constrained triple
                        mct = min(req, key=lambda ti: len(tri2blocks[ti]))
                        cands = tri2blocks[mct]
                        best_ck = -1
                        best_new = 10**9
                        for ck in cands:
                            if ck in cur_set:
                                continue
                            ok = True
                            for ti in req:
                                if not (block_mask[ck] >> ti) & 1:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            new_cnt = 0
                            for ti in block_tris[ck]:
                                if cnt[ti] == 0:
                                    new_cnt += 1
                            if new_cnt < best_new:
                                best_new = new_cnt
                                best_ck = ck
                        if best_ck >= 0:
                            # Remove bi and bj
                            for ti in block_tris[bi]:
                                cnt[ti] -= 1
                            for ti in block_tris[bj]:
                                cnt[ti] -= 1
                            cur_set.discard(bi)
                            cur_set.discard(bj)
                            # Add best_ck
                            for ti in block_tris[best_ck]:
                                cnt[ti] += 1
                            cur_set.add(best_ck)
                            cur[indices[a]] = best_ck
                            cur[indices[b]] = None
                            improved = True
                            found = True
                            break
                    if found:
                        break

            if not cur:
                break

        # Clean up Nones
        cur = [x for x in cur if x is not None]
        return cur

    # Main loop
    best = None
    best_n = 10**9

    for s in range(20):
        if elapsed() > DEADLINE - 20:
            break
        gblocks, gset = run_greedy(s)
        if not gblocks:
            continue
        refined = polish(gblocks)
        n = len(refined)
        if n < best_n:
            best_n = n
            best = refined

    if best is None:
        # Fallback
        gblocks, _ = run_greedy(0)
        best = gblocks
        best_n = len(best)

    return [list(all_blocks[bi]) for bi in best]
