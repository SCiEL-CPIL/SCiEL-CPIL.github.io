import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    rng = _rng(_SEED)

    all_tsubs = list(itertools.combinations(range(v), t))
    n_all = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    count = np.zeros(n_all, dtype=np.int32)
    blocks = []

    def get_block_tsub_indices(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]

    def add_block(block):
        block = tuple(sorted(block))
        blocks.append(block)
        for idx in get_block_tsub_indices(block):
            count[idx] += 1

    def find_best_block(n_starts=30, n_cands=35):
        """Find the best block to add. Returns (block, gain) or (None, 0)."""
        t0 = time.time()
        uncovered_mask = (count == 0)
        n_unc = int(np.sum(uncovered_mask))
        if n_unc == 0:
            return None, 0

        unc_indices = np.where(uncovered_mask)[0]
        n_sample = min(n_starts, len(unc_indices))
        sample_idx = rng.choice(len(unc_indices), size=n_sample, replace=False)

        best_block = None
        best_gain = 0

        for si in sample_idx:
            if time.time() > deadline - 5:
                break
            start_ts = all_tsubs[unc_indices[sample_idx[si]]]
            block = list(start_ts)
            remaining = set(range(v))
            for p in start_ts:
                remaining.discard(p)

            for _ in range(k - t):
                if time.time() > deadline - 5:
                    break
                best_cand = None
                best_s = -1
                rem_list = sorted(remaining)
                n_rem = len(rem_list)
                nc = min(n_cands, n_rem)
                ci_arr = rng.choice(n_rem, size=nc, replace=False)
                for ci in ci_arr:
                    cand = rem_list[ci]
                    b = tuple(sorted(block + [cand]))
                    gain = 0
                    for ts in itertools.combinations(b, t):
                        if count[tsub_to_idx[ts]] == 0:
                            gain += 1
                    if gain > best_s:
                        best_s = gain
                        best_cand = cand
                if best_cand is None:
                    best_cand = remaining.pop()
                block.append(best_cand)
                remaining.discard(best_cand)

            block = tuple(sorted(block))
            gain = 0
            for ts in itertools.combinations(block, t):
                if count[tsub_to_idx[ts]] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        return best_block, best_gain

    # Phase 1: Greedy construction
    while True:
        if time.time() > deadline - 25:
            break
        b, g = find_best_block(n_starts=40, n_cands=40)
        if b is None or g == 0:
            break
        add_block(b)

    # Phase 2: 1-exchange local search
    any_improved = True
    while any_improved:
        any_improved = False
        if time.time() > deadline - 5:
            break
        for bi in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            old = blocks[bi]
            for idx in get_block_tsub_indices(old):
                count[idx] -= 1

            # Find best replacement
            best_repl = None
            best_gain = 0
            uncovered_mask = (count == 0)
            n_unc = int(np.sum(uncovered_mask))
            if n_unc > 0:
                unc_indices = np.where(uncovered_mask)[0]
                n_sample = min(20, len(unc_indices))
                sample_idx = rng.choice(len(unc_indices), size=n_sample, replace=False)
                for si in sample_idx:
                    if time.time() > deadline - 5:
                        break
                    start_ts = all_tsubs[unc_indices[sample_idx[si]]]
                    block = list(start_ts)
                    remaining = set(range(v))
                    for p in start_ts:
                        remaining.discard(p)
                    for _ in range(k - t):
                        if time.time() > deadline - 5:
                            break
                        best_cand = None
                        best_s = -1
                        rem_list = sorted(remaining)
                        n_rem = len(rem_list)
                        nc = min(25, n_rem)
                        ci_arr = rng.choice(n_rem, size=nc, replace=False)
                        for ci in ci_arr:
                            cand = rem_list[ci]
                            b = tuple(sorted(block + [cand]))
                            gain = 0
                            for ts in itertools.combinations(b, t):
                                if count[tsub_to_idx[ts]] == 0:
                                    gain += 1
                            if gain > best_s:
                                best_s = gain
                                best_cand = cand
                        if best_cand is None:
                            best_cand = remaining.pop()
                        block.append(best_cand)
                        remaining.discard(best_cand)
                    block = tuple(sorted(block))
                    gain = 0
                    for ts in itertools.combinations(block, t):
                        if count[tsub_to_idx[ts]] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = block

            if best_repl is not None and best_gain > 0:
                blocks[bi] = best_repl
                for idx in get_block_tsub_indices(best_repl):
                    count[idx] += 1
                any_improved = True
            else:
                for idx in get_block_tsub_indices(old):
                    count[idx] += 1

    # Phase 3: Fill remaining uncovered
    while True:
        if time.time() > deadline - 1:
            break
        if not np.any(count == 0):
            break
        unc_indices = np.where(count == 0)[0]
        if len(unc_indices) == 0:
            break
        tr = all_tsubs[unc_indices[0]]
        block = list(tr)
        remaining = set(range(v))
        for p in tr:
            remaining.discard(p)
        for _ in range(k - t):
            if time.time() > deadline - 1:
                break
            best_cand = None
            best_s = -1
            rem_list = sorted(remaining)
            n_rem = len(rem_list)
            nc = min(25, n_rem)
            if n_rem > 0:
                ci_arr = rng.choice(n_rem, size=nc, replace=False)
                for ci in ci_arr:
                    cand = rem_list[ci]
                    b = tuple(sorted(block + [cand]))
                    gain = 0
                    for ts in itertools.combinations(b, t):
                        if count[tsub_to_idx[ts]] == 0:
                            gain += 1
                    if gain > best_s:
                        best_s = gain
                        best_cand = cand
            if best_cand is None:
                best_cand = remaining.pop()
            block.append(best_cand)
            remaining.discard(best_cand)
        add_block(tuple(sorted(block)))

    return [list(b) for b in blocks]
