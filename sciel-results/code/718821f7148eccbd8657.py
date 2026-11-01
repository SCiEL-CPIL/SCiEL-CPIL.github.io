"""Global block-level greedy set cover + local improvement, vectorized, with
deterministic multi-pass restarts and an explicit wall-clock deadline guard.
"""
import time
import random
import itertools
import numpy as np

# --- precomputation (cached, done once) ---
_CACHE = {}


def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]

    pts = np.arange(v)
    all_k = np.array(list(itertools.combinations(range(v), k)), dtype=np.int64)
    all_t = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    nt = len(all_t)
    nk = len(all_k)

    # membership: (nk, nt) bool, block b covers triple j
    mem = np.zeros((nk, nt), dtype=bool)
    for b in range(nk):
        blk = all_k[b]
        for tr in itertools.combinations(blk, t):
            mem[b, int(np.searchsorted(all_t, tr, side='left'))] = True

    # for each triple, the list of blocks that cover it (for seeding)
    cov = [np.nonzero(mem[:, j])[0] for j in range(nt)]

    _CACHE[key] = (all_k, all_t, mem, cov, pts, nt, nk)
    return _CACHE[key]


def _greedy(all_k, all_t, mem, cov, pts, nt, nk, t, seed, deadline):
    """One global greedy run. Returns (blocks, covered_count)."""
    rng = random.Random(seed)
    # start from the least-covered triple (deterministic), then randomize order
    counts = mem.sum(axis=0)
    start = int(np.argmin(counts))
    uncovered = set(range(nt))
    uncovered.remove(start)

    sel = []
    sel_set = set()
    # randomize the candidate ordering for tie-breaking
    order = list(range(nk))
    rng.shuffle(order)
    rank = {b: i for i, b in enumerate(order)}

    while uncovered:
        # score each candidate block by how many uncovered triples it covers
        best = -1
        best_score = -1
        # vectorized over blocks: sum of coverage over currently-uncovered triples
        unc = np.array(sorted(uncovered), dtype=np.int64)
        scores = mem[:, unc].sum(axis=1)
        # restrict to blocks not already selected
        for b in order:
            if b in sel_set:
                continue
            s = int(scores[b])
            if s > best_score:
                best_score = s
                best = b
                if best_score == len(unc):  # perfect, can't do better
                    break
            # early exit when we can't beat current best with remaining? keep simple
        if best < 0:
            break
        sel.append(best)
        sel_set.add(best)
        # remove all triples covered by this block
        covered_by_block = np.nonzero(mem[best])[0]
        for j in covered_by_block:
            uncovered.discard(int(j))

    blocks = [tuple(all_k[b]) for b in sel]
    covered = nt - len(uncovered)
    return blocks, covered


def _local_improve(all_k, all_t, mem, cov, pts, nt, nk, t, blocks, deadline):
    """Try to reduce block count by replacing one selected block with a better one."""
    if not blocks:
        return blocks
    sel = [list(b) for b in blocks]
    # map tuple -> index in all_k
    idx = {tuple(r): i for i, r in enumerate(all_k)}
    sel_idx = [idx[tuple(b)] for b in sel]
    sel_set = set(sel_idx)

    # current coverage count per triple
    covcnt = mem[sel_idx].sum(axis=0)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # for each selected block, find a replacement that covers at least as many
        # uncovered-after-removal triples
        for pos in range(len(sel_idx)):
            cur = sel_idx[pos]
            # triples that would become uncovered if we remove cur
            only = np.nonzero((covcnt == 1) & mem[cur])[0]
            if len(only) == 0:
                # removing cur leaves nothing uncovered -> we can just drop it
                sel_idx.pop(pos)
                sel_set.discard(cur)
                covcnt = mem[sel_idx].sum(axis=0) if sel_idx else np.zeros(nt, dtype=np.int64)
                improved = True
                break
            # need a replacement covering all of `only`
            # candidate blocks that cover every triple in `only`
            cand_mask = np.ones(nk, dtype=bool)
            for j in only:
                cand_mask &= mem[:, int(j)]
            cand_mask[cur] = False
            cands = np.nonzero(cand_mask)[0]
            if len(cands) == 0:
                continue
            # pick the candidate that covers the most currently-covered triples
            # (i.e. maximizes redundancy), to keep coverage safe
            gains = mem[cands].sum(axis=1)
            best_c = int(cands[int(np.argmax(gains))])
            # verify coverage stays full
            trial = sel_idx.copy()
            trial[pos] = best_c
            newcov = mem[trial].sum(axis=0)
            if (newcov > 0).all():
                sel_idx[pos] = best_c
                sel_set.discard(cur)
                sel_set.add(best_c)
                covcnt = newcov
                improved = True
                break
    return [tuple(all_k[b]) for b in sel_idx]


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under the 150s kill
    all_k, all_t, mem, cov, pts, nt, nk = _precompute(v, k, t)

    best_blocks = None
    best_cov = -1
    best_len = 10**9

    seed = 0
    while time.time() < deadline:
        blocks, covered = _greedy(all_k, all_t, mem, cov, pts, nt, nk, t, seed, deadline)
        if time.time() < deadline:
            blocks = _local_improve(all_k, all_t, mem, cov, pts, nt, nk, t, blocks, deadline)
        if len(blocks) < best_len and covered >= nt:
            best_blocks = blocks
            best_cov = covered
            best_len = len(blocks)
        seed += 1

    if best_blocks is None:
        # fallback: pure deterministic greedy (no local improve) to guarantee validity
        blocks, _ = _greedy(all_k, all_t, mem, cov, pts, nt, nk, t, 0, deadline)
        best_blocks = blocks

    # final safety: ensure we have a full cover; if not, fall back to exhaustive-ish
    if time.time() < deadline - 1.0:
        # verify coverage
        if best_blocks:
            cc = np.zeros(nt, dtype=np.int64)
            idx = {tuple(r): i for i, r in enumerate(all_k)}
            for b in best_blocks:
                cc[mem[idx[tuple(b)]]] += 1
            if (cc > 0).all():
                return [list(b) for b in best_blocks]

    # last-resort: rebuild a guaranteed cover greedily until deadline
    uncovered = set(range(nt))
    blocks = []
    while uncovered and time.time() < deadline - 0.5:
        unc = np.array(sorted(uncovered), dtype=np.int64)
        scores = mem[:, unc].sum(axis=1)
        b = int(np.argmax(scores))
        blocks.append(tuple(all_k[b]))
        for j in np.nonzero(mem[b])[0]:
            uncovered.discard(int(j))
    return [list(b) for b in blocks]
