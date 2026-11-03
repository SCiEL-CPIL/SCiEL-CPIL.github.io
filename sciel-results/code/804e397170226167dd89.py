"""Ternary covering code K_3(7,1): ILS with minimal-cover construction + swap descent."""
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin under the 150s limit


def _time_ok():
    return time.time() < _DEADLINE


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 110.0

    if radius != 1:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n  # 2187
    # Precompute balls as bitsets: ball_bits[i] is a Python int whose bit j is 1 iff
    # word j is within distance 1 of word i.
    ball_bits = [0] * total
    # For each position, the set of words having a given value at that position.
    # pos_val[pos][val] -> int bitmask of words with value `val` at position `pos`.
    pos_val = [[0] * q for _ in range(n)]
    for idx in range(total):
        # compute digits
        t = idx
        digits = [0] * n
        for pos in range(n):
            digits[pos] = t % q
            t //= q
        for pos in range(n):
            pos_val[pos][digits[pos]] |= (1 << idx)

    # ball_bits[i] = (1<<i) OR union over pos of (all words with same digits except pos changed)
    # all words with same digits except pos = (all words) minus (words with same value at pos as i)
    # = all_bits ^ pos_val[pos][digits[i][pos]]
    all_bits = (1 << total) - 1
    for idx in range(total):
        t = idx
        digits = [0] * n
        for pos in range(n):
            digits[pos] = t % q
            t //= q
        b = 1 << idx
        for pos in range(n):
            b |= (all_bits ^ pos_val[pos][digits[pos]])
        ball_bits[idx] = b

    def _is_covered(chosen_bits):
        union = 0
        for c in chosen_bits:
            union |= ball_bits[c]
        return union == all_bits

    def _is_minimal(chosen_list):
        m = len(chosen_list)
        for i in range(m):
            # remove i: is union of others still all?
            union = 0
            for j in range(m):
                if j == i:
                    continue
                union |= ball_bits[chosen_list[j]]
            if union == all_bits:
                return False
        return True

    def _remove_redundant(chosen_list):
        changed = True
        while changed:
            changed = False
            if not _time_ok():
                break
            m = len(chosen_list)
            for i in range(m):
                if not _time_ok():
                    break
                union = 0
                for j in range(m):
                    if j == i:
                        continue
                    union |= ball_bits[chosen_list[j]]
                if union == all_bits:
                    chosen_list.pop(i)
                    changed = True
                    break
        return chosen_list

    def _random_minimal_cover(rng):
        # Build a random minimal cover: keep inserting random words until covered,
        # then remove redundants.
        chosen = []
        in_set = [False] * total
        union = 0
        # random permutation of words
        order = list(range(total))
        rng.shuffle(order)
        for c in order:
            if not _time_ok():
                break
            if union == all_bits:
                break
            chosen.append(c)
            in_set[c] = True
            union |= ball_bits[c]
        if union != all_bits:
            return None
        _remove_redundant(chosen)
        if not _is_minimal(chosen):
            # extra cleanup
            _remove_redundant(chosen)
        return chosen

    def _swap_descent(chosen_list, rng):
        # Try 1-for-1 swaps that reduce size: remove i, if still covered, keep removed.
        # Then optionally add a word that allows removing another? We do: 
        #   For each i: if removing i still covers, remove it (that's a size decrease).
        # But that's just redundancy removal. For true swap: replace i with c where
        #   (chosen - {i} + {c}) is a cover AND has a redundant word to remove.
        # Simpler effective move: try to find c not in set such that adding c makes
        # some existing word redundant, then remove that redundant word (net size same),
        # then try redundancy removal again (net decrease). We'll do:
        #   For each i: for each c not in set (sampled): newset = chosen - {i} + {c}
        #     if newset covers: then try to remove any redundant from newset -> if size < m accept.
        m = len(chosen_list)
        in_set = [False] * total
        for c in chosen_list:
            in_set[c] = True
        improved = True
        while improved:
            improved = False
            if not _time_ok():
                break
            m = len(chosen_list)
            # First: pure redundancy removal
            _remove_redundant(chosen_list)
            m = len(chosen_list)
            # Then: swap attempts
            # Sample candidate c's from the whole space not in set
            # To bound work, limit attempts per iteration
            attempts = 0
            max_attempts = 2000
            for i in range(len(chosen_list)):
                if not _time_ok():
                    break
                attempts += 1
                if attempts > max_attempts:
                    break
                # pick a random c not in set
                # We'll sample a few random c's
                for _ in range(3):
                    if not _time_ok():
                        break
                    c = rng.randrange(total)
                    if in_set[c]:
                        continue
                    # newset = chosen without i plus c
                    # check coverage
                    union = ball_bits[c]
                    ok = True
                    for j in range(len(chosen_list)):
                        if j == i:
                            continue
                        union |= ball_bits[chosen_list[j]]
                        if union == all_bits:
                            ok = True
                            break
                    if union != all_bits:
                        ok = False
                    if not ok:
                        continue
                    # newset covers. Build new list
                    new_list = [chosen_list[j] for j in range(len(chosen_list)) if j != i]
                    new_list.append(c)
                    # try to remove a redundant from new_list
                    new_list = _remove_redundant(new_list)
                    if len(new_list) < len(chosen_list):
                        chosen_list = new_list
                        in_set = [False] * total
                        for c2 in chosen_list:
                            in_set[c2] = True
                        improved = True
                        break
                if improved:
                    break
        return chosen_list

    # Main ILS loop
    rng = random.Random(12345)
    best = None
    best_len = 10**9

    # First get a quick greedy-ish cover to have a starting point
    # Greedy: repeatedly add word covering most uncovered
    # But we'll just use random minimal cover for the first iteration too.
    restarts = 0
    while _time_ok():
        restarts += 1
        cover = _random_minimal_cover(rng)
        if cover is None:
            continue
        cover = _swap_descent(cover, rng)
        if len(cover) < best_len:
            best_len = len(cover)
            best = list(cover)
        # Occasionally perturb best for diversity (optional)
        # We'll just keep going with fresh random covers.

    if best is None:
        # Fallback: full space
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Final validity check
    union = 0
    for c in best:
        union |= ball_bits[c]
    if union != all_bits:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Convert linear indices to lists
    out = []
    for c in best:
        t = c
        digits = [0] * n
        for pos in range(n):
            digits[pos] = t % q
            t //= q
        out.append(digits)
    return out
