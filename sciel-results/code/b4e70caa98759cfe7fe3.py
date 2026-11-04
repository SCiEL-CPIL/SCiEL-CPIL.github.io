"""Greedy + local-search construction for ternary covering codes K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def _ball_masks(q, n, radius):
    """Return a list of frozensets; mask[i] = set of integer ids at Hamming
    distance <= radius from word i."""
    total = q ** n
    # Precompute digit vectors.
    words = []
    for i in range(total):
        x = i
        v = [0] * n
        for j in range(n):
            v[j] = x % q
            x //= q
        words.append(v)

    masks = []
    for i in range(total):
        wi = words[i]
        s = {i}
        # radius 1: single-position changes.
        for pos in range(n):
            base = wi[pos]
            for d in range(q):
                if d == base:
                    continue
                v2 = wi[:]
                v2[pos] = d
                val = 0
                for j in range(n - 1, -1, -1):
                    val = val * q + v2[j]
                s.add(val)
        masks.append(s)
    return masks


def _build_code(q, n, radius, deadline, seed):
    total = q ** n
    masks = _ball_masks(q, n, radius)

    # Precompute the integer id for each word given by a base word + one change.
    def neighbor_ids(w):
        ids = {w}
        for pos in range(n):
            base = w % (q ** (n - pos))  # not used; recompute digits
        # recompute digits
        x = w
        digits = [0] * n
        for j in range(n):
            digits[j] = x % q
            x //= q
        for pos in range(n):
            for d in range(q):
                if d == digits[pos]:
                    continue
                nd = digits[:]
                nd[pos] = d
                val = 0
                for j in range(n - 1, -1, -1, ):
                    val = val * q + nd[j]
                ids.add(val)
        return ids

    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)

    chosen = []
    covered = bytearray(total)  # 0/1
    uncovered = total
    for i in order:
        if time.time() > deadline:
            break
        if covered[i]:
            continue
        # add codeword i
        chosen.append(i)
        for j in masks[i]:
            if not covered[j]:
                covered[j] = 1
                uncovered -= 1
        if uncovered == 0:
            break
    return chosen, covered, uncovered, masks


def _local_search(chosen, covered, uncovered, masks, total, deadline, seed):
    """Try to reduce the number of codewords: (a) drop redundant codewords,
    (b) swap a codeword for one that covers more currently-uncovered words."""
    rng = random.Random(seed + 1000)
    chosen = list(chosen)
    covered = bytearray(covered)
    uncovered = int(uncovered)

    def coverage_count(i):
        c = 0
        for j in masks[i]:
            if not covered[j]:
                c += 1
        return c

    def add(i):
        chosen.append(i)
        for j in masks[i]:
            if not covered[j]:
                covered[j] = 1
                uncovered -= 1

    def remove(i):
        chosen.remove(i)
        for j in masks[i]:
            # check if still covered by another chosen
            still = False
            for k in chosen:
                if j in masks[k]:
                    still = True
                    break
            if not still:
                covered[j] = 0
                uncovered += 1

    # (a) drop redundant codewords (those whose ball is fully covered by others)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline:
                return chosen, covered, uncovered
            w = chosen[idx]
            # is every word in masks[w] covered by some other chosen?
            redundant = True
            for j in masks[w]:
                other = False
                for k in chosen:
                    if k == w:
                        continue
                    if j in masks[k]:
                        other = True
                        break
                if not other:
                    redundant = False
                    break
            if redundant:
                chosen.remove(w)
                improved = True

    # (b) swap: replace a low-coverage codeword with a better one if it helps
    for _ in range(40):
        if time.time() > deadline:
            break
        # pick the codeword with the fewest unique (uncovered) contributions
        best_idx = -1
        best_cov = 10 ** 9
        for idx, w in enumerate(chosen):
            c = coverage_count(w)
            if c < best_cov:
                best_cov = c
                best_idx = idx
        if best_idx < 0 or best_cov > 0:
            # removing this one would create new uncovered words; only swap if
            # a single new codeword covers all of them plus more.
            pass
        # candidate: find word covering most uncovered
        cand = -1
        cand_cov = 0
        # sample candidates for speed
        sample = rng.sample(range(total), min(300, total))
        for i in sample:
            c = coverage_count(i)
            if c > cand_cov:
                cand_cov = c
                cand = i
        if cand < 0 or cand in chosen or cand_cov <= 0:
            break
        # try swapping: remove best_idx, see uncovered set, then add cand
        saved = bytearray(covered)
        saved_uncovered = uncovered
        remove(best_idx)
        # now uncovered increased; will cand cover them?
        # check cand covers all currently uncovered
        ok = True
        for j in range(total):
            if not covered[j] and j not in masks[cand]:
                ok = False
                break
        if ok and cand_cov > 0:
            add(cand)
            # net: same count, but we removed one and added one -> no reduction.
            # This swap only helps if it enables later drops; keep it only if
            # coverage strictly improves after re-dropping.
            # Re-run drop pass.
            dropped = False
            for idx2 in range(len(chosen) - 1, -1, -1):
                w2 = chosen[idx2]
                redundant = True
                for j in masks[w2]:
                    other = False
                    for k in chosen:
                        if k == w2:
                            continue
                        if j in masks[k]:
                            other = True
                            break
                    if not other:
                        redundant = False
                        break
                if redundant:
                    chosen.remove(w2)
                    dropped = True
            if not dropped:
                # revert swap (no benefit)
                chosen.clear()
                covered = saved
                uncovered = saved_uncovered
                break
        else:
            # revert
            chosen.clear()
            covered = saved
            uncovered = saved_uncovered
            break

    return chosen, covered, uncovered


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s
    total = q ** n

    # Baseline: direct-sum tetracode construction (81 words for q=3,n=6,r=1).
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        base = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                base.append(h + list(tail))
    else:
        base = []

    best = None
    best_len = 10 ** 9
    if base:
        best = base
        best_len = len(base)

    # Run greedy with several seeds, then local search on the best.
    seeds = [1, 7, 42, 123, 2024]
    for s in seeds:
        if time.time() > deadline - 5.0:
            break
        chosen, covered, uncovered, masks = _build_code(q, n, radius, deadline - 5.0, s)
        if uncovered > 0:
            # complete the coverage with a final greedy sweep over uncovered
            for i in range(total):
                if time.time() > deadline - 5.0:
                    break
                if not covered[i]:
                    chosen.append(i)
                    for j in masks[i]:
                        if not covered[j]:
                            covered[j] = 1
                            uncovered -= 1
                if uncovered == 0:
                    break
        if uncovered == 0 and len(chosen) < best_len:
            # local search to try to shrink further
            chosen2, covered2, unc2 = _local_search(
                chosen, covered, uncovered, masks, total, deadline - 2.0, s)
            if unc2 == 0 and len(chosen2) < best_len:
                best_len = len(chosen2)
                best = chosen2
            elif len(chosen) < best_len:
                best_len = len(chosen)
                best = chosen

    # Convert integer ids back to words.
    result = []
    for w in best:
        x = w
        v = [0] * n
        for j in range(n):
            v[j] = x % q
            x //= q
        result.append(v)
    return result
# EVOLVE-BLOCK-END
