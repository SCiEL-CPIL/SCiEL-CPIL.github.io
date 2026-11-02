"""Direct-sum family optimizer for K_3(7,1) with deadline guard."""
import itertools
import time


def _hamming_le1(a, b):
    d = 0
    for x, y in zip(a, b):
        if x != y:
            d += 1
            if d > 1:
                return False
    return True


def _build_ball(cw, q, n):
    """All words within Hamming distance 1 of cw (inclusive)."""
    ball = [tuple(cw)]
    for i in range(n):
        orig = cw[i]
        for v in range(q):
            if v == orig:
                continue
            w = list(cw)
            w[i] = v
            ball.append(tuple(w))
    return ball


def _coverage(cw_set, q, n):
    """Set of all words covered by the given codeword set."""
    covered = set()
    for cw in cw_set:
        for w in _build_ball(cw, q, n):
            covered.add(w)
    return covered


def _greedy_cover(candidates, uncovered, q, n):
    """Greedy set-cover: repeatedly pick candidate covering most uncovered."""
    # precompute ball membership as frozenset of tuples
    cand_balls = [set(_build_ball(c, q, n)) for c in candidates]
    chosen = []
    un = set(uncovered)
    # for speed, track counts
    while un:
        best_i = -1
        best_cnt = 0
        for i, b in enumerate(cand_balls):
            cnt = len(b & un)
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
                if best_cnt == len(un):
                    break
        if best_cnt == 0:
            break
        chosen.append(candidates[best_i])
        un -= cand_balls[best_i]
    return chosen


def _remove_redundant(cw_list, q, n, deadline):
    """Try to drop each codeword if coverage remains complete."""
    full = set(itertools.product(range(q), repeat=n))
    cw_set = list(cw_list)
    balls = [set(_build_ball(c, q, n)) for c in cw_set]
    covered = set()
    for b in balls:
        covered |= b
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(cw_set)):
            if time.time() >= deadline:
                break
            # covered without i
            without = set()
            for j in range(len(cw_set)):
                if j != i:
                    without |= balls[j]
            if without == full:
                # remove i
                balls.pop(i)
                cw_set.pop(i)
                covered = without
                changed = True
                break
    return cw_set


def _direct_sum_construct(q, n, radius, deadline):
    # Split into base (4) + tail (n-4). For general n, fall back to plain greedy.
    if n < 4:
        # small: just greedy over everything
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        uncovered = set(itertools.product(range(q), repeat=n))
        chosen = _greedy_cover(all_words, uncovered, q, n)
        chosen = _remove_redundant(chosen, q, n, deadline)
        return [list(c) for c in chosen]

    base_len = 4
    tail_len = n - base_len

    base_all = [list(w) for w in itertools.product(range(q), repeat=base_len)]
    tail_all = [list(w) for w in itertools.product(range(q), repeat=tail_len)]

    # Full word space
    full = set(itertools.product(range(q), repeat=n))

    # Greedy pick base words
    base_uncovered = set(itertools.product(range(q), repeat=base_len))
    base_balls = [set(_build_ball(c, q, base_len)) for c in base_all]
    base_chosen = []
    un_b = set(base_uncovered)
    while un_b:
        best_i = -1
        best_cnt = 0
        for i, b in enumerate(base_balls):
            cnt = len(b & un_b)
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
        if best_cnt == 0:
            break
        base_chosen.append(base_all[best_i])
        un_b -= base_balls[best_i]

    # Greedy pick tail words
    tail_uncovered = set(itertools.product(range(q), repeat=tail_len))
    tail_balls = [set(_build_ball(c, q, tail_len)) for c in tail_all]
    tail_chosen = []
    un_t = set(tail_uncovered)
    while un_t:
        best_i = -1
        best_cnt = 0
        for i, b in enumerate(tail_balls):
            cnt = len(b & un_t)
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
        if best_cnt == 0:
            break
        tail_chosen.append(tail_all[best_i])
        un_t -= tail_balls[best_i]

    # Build direct-sum codeword set
    direct = [b + t for b in base_chosen for t in tail_chosen]

    # Now refine: try to remove redundant direct-sum codewords
    direct = _remove_redundant(direct, q, n, deadline)

    return direct


def construct(q, n, radius):
    start = time.time()
    deadline = start + 120.0  # safety margin under 150s kill

    # For the specific known case q=3, n=7, radius=1, use the direct-sum family
    # optimized. Otherwise fall back to the same routine.
    if q == 3 and n == 7 and radius == 1:
        return _direct_sum_construct(3, 7, 1, deadline)

    # General fallback: greedy over full space then redundant removal
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    uncovered = set(itertools.product(range(q), repeat=n))
    chosen = _greedy_cover(all_words, uncovered, q, n)
    chosen = _remove_redundant(chosen, q, n, deadline)
    return [list(c) for c in chosen]
