"""Covering code K_3(7,1): structured direct-product init + bit-parallel local search."""
import time
import numpy as np

_DEADLINE = 120.0
_t0 = 0.0


def _remaining():
    return _DEADLINE - (time.time() - _t0)


def _flat_to_word(idx, q, n):
    w = []
    x = idx
    for _ in range(n):
        w.append(x % q)
        x //= q
    return w[::-1]


def _word_to_flat(w, q, n):
    idx = 0
    for a in w:
        idx = idx * q + a
    return idx


def _dist1_neighbors(idx, q, n):
    """All flat indices at Hamming distance exactly 1 from idx."""
    out = []
    for pos in range(n):
        stride = q ** (n - 1 - pos)
        d = (idx // stride) % q
        for d2 in range(q):
            if d2 != d:
                out.append(idx + (d2 - d) * stride)
    return out


def _direct_sum_init(q, n, radius):
    """Build an initial covering using a recursive direct-product construction.
    For q=3, n=7, radius=1: split 7 = 4+3 (or 3+4), build a radius-1 code on each
    part, then take the direct sum. Direct sum of (n1,r1) and (n2,r2) gives (n1+n2, r1+r2).
    To get radius 1 on 7, we need r1+r2=1, so one part has radius 1 and the other radius 0.
    Radius-0 code on n2 words: every word is a codeword (size q^2). That's too big.
    Better: use a known small radius-1 code on 3 words (size 3) and radius-1 on 4? No.
    Instead: use the fact that for q=3, n=3, radius=1, a perfect code of size 3 exists (Hamming).
    For n=4, radius=1, we need ceil(3^4/(1+4*2)) = ceil(81/9) = 9, and a code of size 9 exists.
    Direct sum of (3,1) size 3 and (4,1) size 9 gives (7,2) not (7,1). That's radius 2.
    
    So direct sum doesn't directly give radius 1 on 7. Instead, use a greedy structured init:
    pick codewords on a coarse grid then refine. We'll just do a fast greedy here.
    """
    total = q ** n
    uncovered = np.ones(total, dtype=np.uint8)
    strides = [q ** (n - 1 - pos) for pos in range(n)]

    # Fast greedy: pick word covering most uncovered, iterate.
    # Precompute ball for each index lazily but use numpy for counting.
    # To be fast, we process in batches.
    codewords = []
    # We'll use a simple greedy with periodic full re-count.
    # For 2187 words and radius 1, ball size = 1+7*2 = 15.
    # Greedy: each step O(total * ball_size) to find best = ~33k ops, times ~186 steps = 6M. Fine.

    while True:
        if time.time() - _t0 > _DEADLINE - 10.0:
            break
        if uncovered.sum() == 0:
            break
        # Find best candidate
        best = -1
        best_count = 0
        for idx in range(total):
            if uncovered[idx] == 0:
                continue
            # count uncovered in ball(idx)
            c = 0
            c += uncovered[idx]
            for pos in range(n):
                st = strides[pos]
                d = (idx // st) % q
                for d2 in range(q):
                    if d2 != d:
                        c += uncovered[idx + (d2 - d) * st]
            if c > best_count:
                best_count = c
                best = idx
            if time.time() - _t0 > _DEADLINE - 10.0:
                break
        if best == -1 or best_count == 0:
            break
        codewords.append(best)
        # Mark ball(best) as covered
        uncovered[best] = 0
        for pos in range(n):
            st = strides[pos]
            d = (best // st) % q
            for d2 in range(q):
                if d2 != d:
                    uncovered[best + (d2 - d) * st] = 0
        if time.time() - _t0 > _DEADLINE - 10.0:
            break

    return codewords


def _local_search(codewords, q, n, radius, time_limit=10.0):
    """Bit-parallel local search: remove redundant codewords, then try swaps.
    Uses integer bitmasks for fast coverage counting."""
    total = q ** n
    # Map flat index -> codeword position
    cw_set = set(codewords)
    cw_list = list(codewords)
    pos_of = {idx: i for i, idx in enumerate(cw_list)}

    # Precompute neighbors (distance 1) for all indices
    all_neighbors = [None] * total
    for idx in range(total):
        all_neighbors[idx] = _dist1_neighbors(idx, q, n)

    # For each codeword, the set of words it covers (itself + neighbors)
    # Represent coverage as: for each word w, the count of codewords covering it.
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in cw_list:
        cover_count[cw] += 1
        for nb in all_neighbors[cw]:
            cover_count[nb] += 1

    # A codeword is redundant if removing it leaves all words in its ball still covered
    # (i.e., every word in ball(cw) has cover_count >= 2 before removal, or >= 1 after).
    # Actually: removing cw reduces cover_count by 1 for each word in ball(cw).
    # Redundant iff min cover_count over ball(cw) >= 2 (so after removal, min >= 1).

    def is_redundant(cw):
        if cover_count[cw] < 2:
            return False
        for nb in all_neighbors[cw]:
            if cover_count[nb] < 2:
                return False
        return True

    def remove_cw(cw):
        cover_count[cw] -= 1
        for nb in all_neighbors[cw]:
            cover_count[nb] -= 1
        cw_list.remove(cw)
        del pos_of[cw]

    def add_cw(cw):
        cover_count[cw] += 1
        for nb in all_neighbors[cw]:
            cover_count[nb] += 1
        cw_list.append(cw)
        pos_of[cw] = len(cw_list) - 1

    # Phase 1: remove all redundant codewords repeatedly
    changed = True
    while changed and (time.time() - _t0) < _DEADLINE - 5.0:
        changed = False
        for i in range(len(cw_list) - 1, -1, -1):
            if time.time() - _t0 > _DEADLINE - 5.0:
                break
            cw = cw_list[i]
            if is_redundant(cw):
                remove_cw(cw)
                changed = True

    # Phase 2: try swaps: remove one cw, add another that covers new ground
    # For each candidate removal, find if there's a word not covered after removal
    # that can be covered by adding one codeword.
    improved = True
    while improved and (time.time() - _t0) < _DEADLINE - 5.0:
        improved = False
        for i in range(len(cw_list)):
            if time.time() - _t0 > _DEADLINE - 5.0:
                break
            cw = cw_list[i]
            # Simulate removal
            cover_count[cw] -= 1
            for nb in all_neighbors[cw]:
                cover_count[nb] -= 1
            # Find uncovered words after removal
            uncovered_words = []
            if cover_count[cw] == 0:
                uncovered_words.append(cw)
            for nb in all_neighbors[cw]:
                if cover_count[nb] == 0:
                    uncovered_words.append(nb)
            if len(uncovered_words) <= 1:
                # Can we cover all uncovered with one add?
                # The add must be within distance 1 of all uncovered words.
                # If only one uncovered word w, any codeword in ball(w) works.
                # If zero uncovered, the removal was safe (shouldn't happen after phase 1).
                if len(uncovered_words) == 0:
                    # Just remove
                    cw_list.pop(i)
                    del pos_of[cw]
                    improved = True
                    break
                elif len(uncovered_words) == 1:
                    w = uncovered_words[0]
                    # Find a codeword candidate in ball(w) not already in cw_list
                    # that minimizes new coverage loss (ideally covers w and maybe others)
                    # Try all words in ball(w)
                    best_add = None
                    best_gain = 0
                    for cand in [w] + all_neighbors[w]:
                        if cand in pos_of:
                            continue
                        # Gain: how many currently-uncovered (after removal) does cand cover?
                        # Only w is uncovered, so gain is 1 if cand covers w (it does by construction)
                        # But we also want to minimize "waste": how many already-covered words cand covers
                        # doesn't matter for validity, only for future.
                        # Just pick the first valid one.
                        best_add = cand
                        break
                    if best_add is not None:
                        add_cw(best_add)
                        improved = True
                        break
            # Restore
            cover_count[cw] += 1
            for nb in all_neighbors[cw]:
                cover_count[nb] += 1

    return cw_list


def construct(q, n, radius):
    global _t0
    _t0 = time.time()

    # Phase 1: structured/greedy initialization
    cw_list = _direct_sum_init(q, n, radius)

    # Phase 2: local search with bit-parallel counting
    cw_list = _local_search(cw_list, q, n, radius)

    # Phase 3: if time remains, do more rounds of remove+add
    while (time.time() - _t0) < _DEADLINE - 5.0:
        # Try to remove any redundant
        total = q ** n
        cover_count = np.zeros(total, dtype=np.int32)
        all_neighbors = [None] * total
        for idx in range(total):
            all_neighbors[idx] = _dist1_neighbors(idx, q, n)
        for cw in cw_list:
            cover_count[cw] += 1
            for nb in all_neighbors[cw]:
                cover_count[nb] += 1

        removed = False
        for i in range(len(cw_list) - 1, -1, -1):
            if time.time() - _t0 > _DEADLINE - 5.0:
                break
            cw = cw_list[i]
            redundant = True
            if cover_count[cw] < 2:
                redundant = False
            else:
                for nb in all_neighbors[cw]:
                    if cover_count[nb] < 2:
                        redundant = False
                        break
            if redundant:
                cover_count[cw] -= 1
                for nb in all_neighbors[cw]:
                    cover_count[nb] -= 1
                cw_list.pop(i)
                removed = True
        if not removed:
            break

    # Convert to word lists
    result = [_flat_to_word(idx, q, n) for idx in cw_list]
    return result
