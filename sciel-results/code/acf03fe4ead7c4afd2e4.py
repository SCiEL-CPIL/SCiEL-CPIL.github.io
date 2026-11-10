# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _triple_index(v, t):
    # map triple -> index for O(1) lookup
    idx = {}
    for i, ts in enumerate(itertools.combinations(range(v), t)):
        idx[ts] = i
    return idx


def _block_triples(block, t):
    return list(itertools.combinations(block, t))


def _greedy_cover(v, k, t, all_t, t_index, rng, deadline):
    """Randomized greedy set cover. Returns list of blocks (tuples)."""
    n_t = len(all_t)
    uncov = [1] * n_t  # 1 if uncovered
    # precompute for each point pair (i,j) the list of triples containing both? 
    # Actually we need uncov[i,x] = number of uncovered triples containing both i and x.
    # Precompute triples_by_pair: (i,j) -> list of triple indices containing i and j.
    triples_by_pair = defaultdict(list)
    for idx, ts in enumerate(all_t):
        for a in range(t):
            for b in range(a + 1, t):
                key = (ts[a], ts[b])
                triples_by_pair[key].append(idx)
    # We'll maintain uncov_pair[(i,j)] = count of uncovered triples containing i and j.
    uncov_pair = {}
    for key, lst in triples_by_pair.items():
        uncov_pair[key] = len(lst)

    def cover_triples(block):
        for idx in set(_block_triples(block, t)):
            if uncov[idx]:
                uncov[idx] = 0
                for a in range(t):
                    for b in range(a + 1, t):
                        key = (block[a], block[b])
                        uncov_pair[key] -= 1

    blocks = []
    while True:
        if time.time() > deadline:
            break
        if all(c == 0 for c in uncov):
            break
        # pick the uncovered triple with... just first uncovered (or random among uncovered)
        # find an uncovered triple
        first_uncov = None
        for idx in range(n_t):
            if uncov[idx]:
                first_uncov = idx
                break
        if first_uncov is None:
            break
        block = list(all_t[first_uncov])
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for i in block:
                    if i > x:
                        key = (x, i)
                    else:
                        key = (i, x)
                    gain += uncov_pair.get(key, 0)
                # small random tie-break
                gain += rng.random() * 1e-9
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        cover_triples(block)
    return blocks


def _is_valid(blocks, v, t, all_t, t_index):
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return all(ts in covered for ts in all_t)


def _local_search(blocks, v, k, t, all_t, t_index, rng, deadline):
    """Prune redundant blocks and try point swaps to reduce essential-triple count."""
    blocks = [list(b) for b in blocks]
    # count coverage of each triple
    n_t = len(all_t)
    cov_count = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov_count[t_index[ts]] += 1

    def recompute_cov():
        for i in range(n_t):
            cov_count[i] = 0
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_count[t_index[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pass 1: remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for bi in range(len(blocks) - 1, -1, -1):
                b = blocks[bi]
                # check if any triple in b is covered only by this block
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov_count[t_index[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    # remove it
                    for ts in itertools.combinations(b, t):
                        cov_count[t_index[ts]] -= 1
                    blocks.pop(bi)
                    changed = True
                    improved = True
                    break
        # Pass 2: try point swaps
        for bi in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[bi]
            bset = set(b)
            # for each point in b, try replacing with a point not in b
            for pi in range(k):
                old = b[pi]
                # triples in b containing old
                old_triples = []
                for ts in itertools.combinations(b, t):
                    if old in ts:
                        old_triples.append(ts)
                for new in range(v):
                    if new in bset:
                        continue
                    # compute gain: number of old_triples that become non-unique,
                    # minus number of new triples that become unique
                    gain = 0
                    new_triples = []
                    for ts in old_triples:
                        if cov_count[t_index[ts]] == 1:
                            gain += 1
                    nb = b[:pi] + [new] + b[pi + 1:]
                    for ts in itertools.combinations(nb, t):
                        if new in ts:
                            if cov_count[t_index[ts]] == 0:
                                gain -= 1
                            new_triples.append(ts)
                    if gain > 0:
                        # apply
                        for ts in old_triples:
                            cov_count[t_index[ts]] -= 1
                        for ts in new_triples:
                            cov_count[t_index[ts]] += 1
                        b[pi] = new
                        bset.remove(old)
                        bset.add(new)
                        b.sort()
                        blocks[bi] = b
                        improved = True
                        break
                if improved:
                    break
    return [tuple(sorted(b)) for b in blocks]


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    all_t = _all_t_subsets(v, t)
    t_index = _triple_index(v, t)

    best_blocks = None
    best_len = float('inf')

    # First a deterministic greedy
    rng0 = random.Random(0)
    blocks0 = _greedy_cover(v, k, t, all_t, t_index, rng0, deadline)
    if blocks0:
        blocks0 = _local_search(blocks0, v, k, t, all_t, t_index, rng0, deadline)
        if len(blocks0) < best_len:
            best_len = len(blocks0)
            best_blocks = blocks0

    # Randomized restarts
    seed = 1
    while time.time() < deadline - 5.0:
        rng = random.Random(seed)
        blocks = _greedy_cover(v, k, t, all_t, t_index, rng, deadline)
        if blocks:
            blocks = _local_search(blocks, v, k, t, all_t, t_index, rng, deadline)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
        seed += 1

    if best_blocks is None:
        # fallback: pure greedy without local search
        rng = random.Random(42)
        best_blocks = _greedy_cover(v, k, t, all_t, t_index, rng, deadline)

    # final validity check
    if not _is_valid(best_blocks, v, t, all_t, t_index):
        # rebuild a valid one
        rng = random.Random(99)
        best_blocks = _greedy_cover(v, k, t, all_t, t_index, rng, deadline)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
