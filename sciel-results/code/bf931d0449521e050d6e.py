"""Vectorized NumPy greedy covering with restarts and redundancy removal."""
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

_DEADLINE = 150.0
_MARGIN = 5.0

def _tsub_to_idx(v, t):
    subs = list(combinations(range(v), t))
    arr = np.array(subs, dtype=np.int64)
    d = {s: i for i, s in enumerate(subs)}
    return arr, d

def _redundant_blocks(blocks, t, deadline):
    n = len(blocks)
    if n == 0:
        return blocks
    block_tsubs = [list(combinations(b, t)) for b in blocks]
    tsub_coverers = defaultdict(set)
    for bi, bts in enumerate(block_tsubs):
        for ts in bts:
            tsub_coverers[ts].add(bi)
    keep = [True] * n
    for bi in range(n):
        if not keep[bi]:
            continue
        if time.time() > deadline:
            break
        redundant = True
        for ts in block_tsubs[bi]:
            if len(tsub_coverers[ts]) == 1:
                redundant = False
                break
        if redundant:
            keep[bi] = False
    return [blocks[i] for i in range(n) if keep[i]]

def _verify(blocks, tsub_idx, tsub_d):
    N = tsub_idx.shape[0]
    covered = np.zeros(N, dtype=bool)
    for b in blocks:
        for ts in combinations(b, t):
            covered[tsub_d[ts]] = True
    return covered.all()

def _greedy_once(v, k, t, tsub_idx, tsub_d, rng, deadline):
    N = tsub_idx.shape[0]
    covered = np.zeros(N, dtype=bool)
    blocks = []
    while True:
        if time.time() > deadline:
            return None
        if covered.all():
            break
        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break
        seed = tsub_idx[rng.choice(unc_idx)]
        block = list(seed)
        while len(block) < k:
            if time.time() > deadline:
                return None
            block_mask = np.zeros(v, dtype=bool)
            for b in block:
                block_mask[b] = True
            candidates = [x for x in range(v) if not block_mask[x]]
            cand_arr = np.array(candidates, dtype=np.int64)
            in_block = np.zeros((N, t), dtype=bool)
            for i in range(t):
                in_block[:, i] = block_mask[tsub_idx[:, i]]
            not_in_block = ~in_block
            C = len(cand_arr)
            diff = tsub_idx[:, :, None] != cand_arr[None, None, :]
            not_in_block_ext = not_in_block[:, :, None]
            bad = (not_in_block_ext & diff).any(axis=1)
            gain = (~bad & (~covered)[:, None]).sum(axis=0)
            if gain.max() > 0:
                max_g = gain.max()
                best_candidates = np.where(gain == max_g)[0]
                best_j = rng.choice(best_candidates)
                best_x = int(cand_arr[best_j])
            else:
                best_x = int(cand_arr[rng.choice(C)])
            block.append(best_x)
        block.sort()
        blocks.append(block)
        for ts in combinations(block, t):
            covered[tsub_d[ts]] = True
    return blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    tsub_idx, tsub_d = _tsub_to_idx(v, t)
    N = tsub_idx.shape[0]
    best = None
    best_count = None
    max_restarts = 200
    rng_seed = 12345
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(rng_seed + restart)
        blocks = _greedy_once(v, k, t, tsub_idx, tsub_d, rng, deadline)
        if blocks is None:
            break
        if not blocks:
            continue
        if not _verify(blocks, tsub_idx, tsub_d):
            continue
        if time.time() > deadline:
            break
        blocks = _redundant_blocks(blocks, t, deadline)
        if not _verify(blocks, tsub_idx, tsub_d):
            continue
        cnt = len(blocks)
        if best_count is None or cnt < best_count:
            best_count = cnt
            best = blocks
        if best_count <= 140 and restart >= 5:
            break
    if best is None:
        rng = np.random.default_rng(0)
        blocks = _greedy_once(v, k, t, tsub_idx, tsub_d, rng, deadline)
        if blocks is not None:
            if _verify(blocks, tsub_idx, tsub_d):
                best = blocks
            else:
                best = blocks
        else:
            best = []
            covered = np.zeros(N, dtype=bool)
            for s in range(N):
                if covered[s]:
                    continue
                block = list(tsub_idx[s])
                while len(block) < k:
                    for x in range(v):
                        if x not in block:
                            block.append(x)
                            break
                block.sort()
                best.append(block)
                for ts in combinations(block, t):
                    covered[tsub_d[ts]] = True
            best = _redundant_blocks(best, t, deadline)
    return best
