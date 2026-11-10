# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def _greedy_cover(v, k, t, rng, deadline):
    """Greedy set cover with randomization in seed/tie-breaking."""
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    uncovered = bytearray([1]) * n_t
    blocks = []

    # Precompute for each point, the list of t-subset indices containing it
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_t):
        for p in ts:
            point_to_ts[p].append(i)

    n_uncovered = n_t
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        # Pick a seed t-subset: among uncovered, choose one minimizing number of
        # uncovered t-subsets that intersect it (harder to cover). Random tie-break.
        best_seed = None
        best_score = None
        candidates = [i for i in range(n_t) if uncovered[i]]
        if not candidates:
            break
        if len(candidates) > 200:
            sample = rng.sample(candidates, 200)
        else:
            sample = candidates
        for i in sample:
            ts = all_t[i]
            s = 0
            seen = set()
            for p in ts:
                for j in point_to_ts[p]:
                    if j != i and uncovered[j] and j not in seen:
                        seen.add(j)
                        s += 1
            if best_score is None or s < best_score:
                best_score = s
                best_seed = i
            elif s == best_score and rng.random() < 0.3:
                best_seed = i
        if best_seed is None:
            best_seed = candidates[0]
        block = list(all_t[best_seed])
        # grow block to size k
        while len(block) < k:
            in_block = set(block)
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in in_block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_index.get(ts)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and gain > 0 and rng.random() < 0.2:
                    best_x = x
            if best_x is None or best_gain <= 0:
                unused = [x for x in range(v) if x not in in_block]
                if not unused:
                    break
                best_x = rng.choice(unused)
            block.append(best_x)
        block.sort()
        blocks.append(block)
        # mark covered
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if uncovered[idx]:
                uncovered[idx] = 0
                n_uncovered -= 1
    return blocks


def _is_valid(blocks, v, k, t):
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == len(list(itertools.combinations(range(v), t)))


def _local_search(blocks, v, k, t, rng, deadline):
    """Try to reduce block count via exchange moves with incremental coverage."""
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    blocks = [sorted(b) for b in blocks]

    # coverage count per t-subset
    cov = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1

    def add_block_cov(b):
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1

    def remove_block_cov(b):
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] -= 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        for bi in order:
            if time.time() > deadline:
                return blocks
            if bi >= len(blocks):
                continue
            b = blocks[bi]
            # t-subsets uniquely covered by b
            unique = []
            for ts in itertools.combinations(b, t):
                idx = t_index[ts]
                if cov[idx] == 1:
                    unique.append(idx)
            if not unique:
                # b is fully redundant
                remove_block_cov(b)
                blocks.pop(bi)
                improved = True
                break
            # try to cover unique t-subsets with one new block
            cand = set()
            for idx in unique:
                cand.update(all_t[idx])
            cand = list(cand)
            if len(cand) < k:
                continue
            # find a k-subset of cand covering all unique
            remaining = set(unique)
            newblock = []
            for _ in range(k):
                if not remaining:
                    break
                best_x = None
                best_c = -1
                for x in cand:
                    if x in newblock:
                        continue
                    c = 0
                    for idx in remaining:
                        if x in all_t[idx]:
                            c += 1
                    if c > best_c:
                        best_c = c
                        best_x = x
                if best_x is None or best_c <= 0:
                    break
                newblock.append(best_x)
                to_remove = set()
                for idx in remaining:
                    if all(p in newblock for p in all_t[idx]):
                        to_remove.add(idx)
                remaining -= to_remove
            if not remaining:
                newblock.sort()
                if newblock not in blocks:
                    remove_block_cov(b)
                    blocks.pop(bi)
                    blocks.append(newblock)
                    add_block_cov(newblock)
                    improved = True
                    break
        # 2-block exchange: try replacing two blocks with one
        if time.time() < deadline and len(blocks) >= 2:
            n = len(blocks)
            pairs = list(itertools.combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if time.time() > deadline:
                    break
                if i >= len(blocks) or j >= len(blocks):
                    continue
                bi, bj = blocks[i], blocks[j]
                # t-subsets covered only by bi or bj
                unique_ij = []
                for ts in itertools.combinations(bi, t):
                    idx = t_index[ts]
                    if cov[idx] == 1:
                        unique_ij.append(idx)
                for ts in itertools.combinations(bj, t):
                    idx = t_index[ts]
                    if cov[idx] == 1:
                        unique_ij.append(idx)
                if not unique_ij:
                    # both redundant
                    remove_block_cov(bi)
                    remove_block_cov(bj)
                    blocks.pop(max(i, j))
                    blocks.pop(min(i, j))
                    improved = True
                    break
                # try to cover unique_ij with one new block
                cand = set()
                for idx in unique_ij:
                    cand.update(all_t[idx])
                cand = list(cand)
                if len(cand) < k:
                    continue
                remaining = set(unique_ij)
                newblock = []
                for _ in range(k):
                    if not remaining:
                        break
                    best_x = None
                    best_c = -1
                    for x in cand:
                        if x in newblock:
                            continue
                        c = 0
                        for idx in remaining:
                            if x in all_t[idx]:
                                c += 1
                        if c > best_c:
                            best_c = c
                            best_x = x
                    if best_x is None or best_c <= 0:
                        break
                    newblock.append(best_x)
                    to_remove = set()
                    for idx in remaining:
                        if all(p in newblock for p in all_t[idx]):
                            to_remove.add(idx)
                    remaining -= to_remove
                if not remaining:
                    newblock.sort()
                    if newblock not in blocks:
                        remove_block_cov(bi)
                        remove_block_cov(bj)
                        blocks.pop(max(i, j))
                        blocks.pop(min(i, j))
                        blocks.append(newblock)
                        add_block_cov(newblock)
                        improved = True
                        break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0  # safety margin before 120s kill
    best_blocks = None
    best_len = None
    rng = random.Random(12345)

    # First a deterministic-ish greedy to get a baseline
    b0 = _greedy_cover(v, k, t, rng, deadline)
    if _is_valid(b0, v, k, t):
        best_blocks = b0
        best_len = len(b0)

    # local search on baseline
    if best_blocks is not None and time.time() < deadline:
        ls = _local_search(best_blocks, v, k, t, rng, deadline)
        if len(ls) < best_len:
            best_blocks = ls
            best_len = len(ls)

    # randomized restarts with local search
    attempt = 0
    while time.time() < deadline - 15.0:
        attempt += 1
        rng2 = random.Random(1000 + attempt)
        b = _greedy_cover(v, k, t, rng2, deadline - 12.0)
        if not _is_valid(b, v, k, t):
            continue
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
        # local search
        if time.time() < deadline - 8.0:
            ls = _local_search(b, v, k, t, rng2, deadline - 8.0)
            if len(ls) < best_len:
                best_len = len(ls)
                best_blocks = ls

    if best_blocks is None:
        # fallback: exhaustive is impossible; return a valid greedy
        b = _greedy_cover(v, k, t, random.Random(1), time.time() + 5)
        return b
    return best_blocks
# EVOLVE-BLOCK-END
