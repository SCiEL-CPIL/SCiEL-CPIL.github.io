"""Football pool covering code K_3(7,1).
Strategy: bitset-based greedy init + aggressive local search (remove + swap).
Uses Python integers as bitsets for fast coverage accounting.
"""
import time
import itertools
import random

_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin before kill


def construct(q, n, radius):
    start = time.time()
    total = q ** n  # 2187 for q=3, n=7

    # Precompute balls as frozensets and as bitsets (Python int with bits set)
    # digits_of: integer -> list of n digits
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d

    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * q

    # Precompute ball for each word
    balls = []  # list of frozensets
    ball_bits = []  # list of int bitsets
    for i in range(total):
        d = digits_of(i)
        s = {i}
        for pos in range(n):
            cur = d[pos]
            for v in range(q):
                if v == cur:
                    continue
                s.add(i + (v - cur) * power[pos])
        balls.append(frozenset(s))
        # build bitset
        bits = 0
        for w in s:
            bits |= (1 << w)
        ball_bits.append(bits)

    # Greedy initialization using bitsets
    uncovered_mask = (1 << total) - 1  # all bits set
    selected = []
    while uncovered_mask:
        best_idx = -1
        best_cnt = -1
        for c in range(total):
            cnt = bin(uncovered_mask & ball_bits[c]).count('1')
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = c
                if cnt == 0:
                    break
        if best_idx < 0 or best_cnt == 0:
            break
        selected.append(best_idx)
        uncovered_mask &= ~ball_bits[best_idx]
        if time.time() > start + _DEADLINE - _MARGIN:
            break

    # Local search: iterative removal + swap
    # Maintain coverage counts as a list of ints
    def build_cov(sel):
        cov = [0] * total
        for c in sel:
            for w in balls[c]:
                cov[w] += 1
        return cov

    cov = build_cov(selected)
    best_list = list(selected)
    best_size = len(best_list)

    def is_valid(cov_arr):
        return min(cov_arr) >= 1

    improved = True
    while improved and (time.time() < start + _DEADLINE - _MARGIN):
        improved = False
        # Pass 1: remove redundant codewords
        for k in range(len(selected) - 1, -1, -1):
            c = selected[k]
            # Check if any word in ball(c) has cov==1
            redundant = True
            for w in balls[c]:
                if cov[w] == 1:
                    redundant = False
                    break
            if redundant:
                for w in balls[c]:
                    cov[w] -= 1
                selected.pop(k)
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_list = list(selected)

        # Pass 2: swaps
        if time.time() >= start + _DEADLINE - _MARGIN:
            break
        # Rebuild cov if it changed
        cov = build_cov(selected)
        sel_set = set(selected)
        for i in range(len(selected)):
            if time.time() > start + _DEADLINE - _MARGIN:
                break
            ci = selected[i]
            # Find words uniquely covered by ci
            uniq = [w for w in balls[ci] if cov[w] == 1]
            if not uniq:
                continue
            # Try to find a replacement j that covers all unique words
            for j in range(total):
                if j in sel_set:
                    continue
                # Check j covers all unique words
                ok = True
                for w in uniq:
                    if not (ball_bits[j] >> w) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                # Perform swap
                new_sel = list(selected)
                new_sel[i] = j
                ncov = build_cov(new_sel)
                # Try removals
                changed = True
                while changed:
                    changed = False
                    for k in range(len(new_sel) - 1, -1, -1):
                        c2 = new_sel[k]
                        redundant = True
                        for w in balls[c2]:
                            if ncov[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            for w in balls[c2]:
                                ncov[w] -= 1
                            new_sel.pop(k)
                            changed = True
                if len(new_sel) < len(selected):
                    selected = new_sel
                    cov = ncov
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_list = list(selected)
                    break
                # else try next j

    # Final validity check
    cov = build_cov(selected)
    if min(cov) < 1:
        # Greedy completion
        uncovered = [w for w in range(total) if cov[w] < 1]
        while uncovered:
            best_idx = -1
            best_cnt = -1
            unc_set = set(uncovered)
            for c in range(total):
                cnt = len(unc_set & balls[c])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = c
            if best_idx < 0 or best_cnt == 0:
                break
            selected.append(best_idx)
            for w in balls[best_idx]:
                if w in unc_set:
                    unc_set.discard(w)
            uncovered = list(unc_set)
            if time.time() > start + _DEADLINE - _MARGIN:
                break

    # Deduplicate and convert
    seen = set()
    final = []
    for c in selected:
        if c in seen:
            continue
        seen.add(c)
        final.append(digits_of(c))

    if not final:
        # Fallback: exhaustive (should not happen)
        return [list(x) for x in itertools.product(range(q), repeat=n)]

    return final
