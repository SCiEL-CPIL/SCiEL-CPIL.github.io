import time
import random
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start = time.time()
    deadline = start + 114.0

    def _ok():
        return time.time() < deadline

    # Generate all t-subsets
    t_subs = list(combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {s: i for i, s in enumerate(t_subs)}

    # Generate all k-blocks
    all_blocks = list(combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build block->t-subset coverage as bitmasks
    n_words = (n_t + 63) // 64
    block_mask = np.zeros((n_blocks, n_words), dtype=np.uint64)

    for bi, block in enumerate(all_blocks):
        for sub in combinations(block, t):
            idx = sub_index[sub]
            block_mask[bi, idx >> 6] |= np.uint64(1) << (idx & 63)

    if not _ok():
        return []

    # sub_to_blocks: for each t-subset, list of blocks covering it
    sub_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_blocks):
        for w in range(n_words):
            word = int(block_mask[bi, w])
            while word:
                bit = word & (-word)
                pos = (w << 6) + (bit.bit_length() - 1)
                if pos < n_t:
                    sub_to_blocks[pos].append(bi)
                word ^= bit

    if not _ok():
        return []

    def _update_counts(counts, bi, delta):
        """Add or subtract block bi from counts."""
        for w in range(n_words):
            word = int(block_mask[bi, w])
            while word:
                bit = word & (-word)
                pos = (w << 6) + (bit.bit_length() - 1)
                if pos < n_t:
                    counts[pos] += delta
                word ^= bit

    def _prune(selected, max_rounds=10):
        if not selected:
            return selected
        sset = set(selected)
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in sset:
            _update_counts(counts, bi, 1)
        for _ in range(max_rounds):
            if not _ok():
                break
            changed = False
            for bi in list(sset):
                redundant = True
                for w in range(n_words):
                    word = int(block_mask[bi, w])
                    while word:
                        bit = word & (-word)
                        pos = (w << 6) + (bit.bit_length() - 1)
                        if pos < n_t and counts[pos] <= 1:
                            redundant = False
                            break
                        word ^= bit
                    if not redundant:
                        break
                if redundant:
                    sset.discard(bi)
                    _update_counts(counts, bi, -1)
                    changed = True
            if not changed:
                break
        return list(sset)

    def _greedy(randomized=False, seed=None):
        if seed is not None:
            random.seed(seed)
        selected = []
        sset = set()
        # uncovered bitmask
        unc = np.ones(n_words, dtype=np.uint64)
        if (n_t & 63) != 0:
            unc[-1] = np.uint64((1 << (n_t & 63)) - 1)
        unc_count = n_t

        while unc_count > 0 and _ok():
            # Sample candidates
            n_samp = 300
            if randomized:
                cands = random.sample(range(n_blocks), min(n_samp, n_blocks))
            else:
                step = max(1, n_blocks // n_samp)
                cands = list(range(0, n_blocks, step))

            best_b = -1
            best_gain = 0
            for bi in cands:
                if bi in sset:
                    continue
                gain = 0
                for w in range(n_words):
                    gain += int(block_mask[bi, w] & unc[w]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = bi
                    if best_gain == 56:
                        break

            if best_b == -1:
                break

            selected.append(best_b)
            sset.add(best_b)
            for w in range(n_words):
                unc[w] &= ~block_mask[best_b, w]
            unc_count -= best_gain

        return selected

    def _local_search(selected, max_steps=2000):
        if not selected:
            return selected
        sset = set(selected)
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in sset:
            _update_counts(counts, bi, 1)

        best_sset = set(sset)
        best_sz = len(best_sset)

        for _ in range(max_steps):
            if not _ok():
                break
            if not best_sset:
                break
            bi = random.choice(list(best_sset))
            # Check uncovered if removed
            unc_list = []
            for w in range(n_words):
                word = int(block_mask[bi, w])
                while word:
                    bit = word & (-word)
                    pos = (w << 6) + (bit.bit_length() - 1)
                    if pos < n_t and counts[pos] <= 1:
                        unc_list.append(pos)
                    word ^= bit

            if not unc_list:
                best_sset.discard(bi)
                _update_counts(counts, bi, -1)
                if len(best_sset) < best_sz:
                    best_sz = len(best_sset)
                continue

            # Find replacement
            target = min(unc_list, key=lambda s: len(sub_to_blocks[s]))
            cands = sub_to_blocks[target]
            if not cands:
                continue
            sample = random.sample(cands, min(40, len(cands)))
            repl = -1
            for rb in sample:
                if rb in best_sset:
                    continue
                ok = True
                for s in unc_list:
                    if not (int(block_mask[rb, s >> 6]) >> (s & 63)) & 1:
                        ok = False
                        break
                if ok:
                    repl = rb
                    break

            if repl != -1:
                best_sset.discard(bi)
                _update_counts(counts, bi, -1)
                best_sset.add(repl)
                _update_counts(counts, repl, 1)
                if len(best_sset) < best_sz:
                    best_sz = len(best_sset)

        return list(best_sset)

    def _kick_search(selected, max_steps=4000):
        if not selected:
            return selected
        sset = set(selected)
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in sset:
            _update_counts(counts, bi, 1)

        best_sset = set(sset)
        best_sz = len(best_sset)

        for _ in range(max_steps):
            if not _ok():
                break
            if not sset:
                break

            if random.random() < 0.65:
                # Try removal
                bi = random.choice(list(sset))
                unc_list = []
                for w in range(n_words):
                    word = int(block_mask[bi, w])
                    while word:
                        bit = word & (-word)
                        pos = (w << 6) + (bit.bit_length() - 1)
                        if pos < n_t and counts[pos] <= 1:
                            unc_list.append(pos)
                        word ^= bit

                if not unc_list:
                    sset.discard(bi)
                    _update_counts(counts, bi, -1)
                    if len(sset) < best_sz:
                        best_sz = len(sset)
                        best_sset = set(sset)
                elif random.random() < 0.08:
                    # Kick
                    sset.discard(bi)
                    _update_counts(counts, bi, -1)
                    if len(sset) < best_sz:
                        best_sz = len(sset)
                        best_sset = set(sset)
            else:
                # Add block covering uncovered
                # Find uncovered
                found = False
                for w in range(n_words):
                    # Check if any bit in counts is 0
                    word = int(counts[w])
                    if word != 0xFFFFFFFFFFFFFFFF:  # not all covered in this word
                        # Find a specific uncovered position
                        for b in range(64):
                            pos = (w << 6) + b
                            if pos < n_t and counts[pos] == 0:
                                cands = sub_to_blocks[pos]
                                if cands:
                                    sample = random.sample(cands, min(20, len(cands)))
                                    best_b = sample[0]
                                    best_gain = 0
                                    for cb in sample:
                                        gain = 0
                                        for ww in range(n_words):
                                            gain += int(block_mask[cb, ww] & ~np.uint64(counts[ww])).bit_count()
                                        if gain > best_gain:
                                            best_gain = gain
                                            best_b = cb
                                    sset.add(best_b)
                                    _update_counts(counts, best_b, 1)
                                    found = True
                                break
                    if found:
                        break
                if found:
                    continue

        # Repair: ensure full coverage
        final_counts = np.zeros(n_t, dtype=np.int32)
        for bi in best_sset:
            _update_counts(final_counts, bi, 1)

        for s in range(n_t):
            if final_counts[s] == 0:
                cands = sub_to_blocks[s]
                if cands:
                    best_b = cands[0]
                    best_gain = 0
                    for cb in random.sample(cands, min(15, len(cands))):
                        gain = 0
                        for ww in range(n_words):
                            gain += int(block_mask[cb, ww] & ~np.uint64(final_counts[ww])).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_b = cb
                    best_sset.add(best_b)
                    _update_counts(final_counts, best_b, 1)

        return list(best_sset)

    # Main
    best_result = None
    best_size = float('inf')

    # Deterministic greedy
    if _ok():
        sel = _greedy(randomized=False)
        sel = _prune(sel)
        sel = _local_search(sel, 1500)
        if len(sel) < best_size:
            best_size = len(sel)
            best_result = sel

    # Randomized restarts
    for r in range(10):
        if not _ok():
            break
        sel = _greedy(randomized=True, seed=100 + r)
        sel = _prune(sel)
        sel = _local_search(sel, 800)
        if len(sel) < best_size:
            best_size = len(sel)
            best_result = sel

    # Kick search on best
    if _ok() and best_result:
        sel = _kick_search(best_result, 3000)
        if len(sel) < best_size:
            best_size = len(sel)
            best_result = sel

    # Final polish
    if _ok() and best_result:
        sel = _local_search(best_result, 2000)
        if len(sel) < best_size:
            best_size = len(sel)
            best_result = sel
        sel = _prune(best_result, 15)
        if len(sel) < best_size:
            best_size = len(sel)
            best_result = sel

    if best_result is None:
        best_result = _greedy(randomized=False)
        best_result = _prune(best_result)

    return [list(all_blocks[i]) for i in best_result]
