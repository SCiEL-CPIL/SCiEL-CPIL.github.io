"""Ternary covering code K_3(7,1) via bit-parallel greedy + aggressive local search.

Structural change from the array-based seed: represent every ball and the covered
set as Python arbitrary-precision integers (bitsets). Ball union/intersection and
"does this ball cover any uncovered word" become O(1) integer ops, which lets us run
many more local-search iterations (deletion, 1-for-1 swaps, 2-for-1 exchanges) and
randomized restarts inside the wall-clock budget.
"""
import time
import random
import numpy as np

# Evaluator kills at 150s; keep a hard safety margin.
DEADLINE = 142.0


def _digits_from_index(idx, n, q):
    d = []
    for _ in range(n):
        d.append(idx % q)
        idx //= q
    return tuple(d)


def _index_from_digits(digits, q):
    idx = 0
    for i, v in enumerate(digits):
        idx += v * (q ** i)
    return idx


def _precompute(q, n, radius):
    total = q ** n
    digits = [_digits_from_index(i, n, q) for i in range(total)]

    ball_list = []
    ball_bits = []
    for i in range(total):
        d = digits[i]
        members = [i]
        for pos in range(n):
            cur = d[pos]
            for nv in range(q):
                if nv == cur:
                    continue
                nd = list(d)
                nd[pos] = nv
                members.append(_index_from_digits(nd, q))
        bl = np.array(members, dtype=np.int64)
        bits = 0
        for m in members:
            bits |= (1 << m)
        ball_list.append(bl)
        ball_bits.append(bits)
    return ball_bits, ball_list


def _precompute_covers(q, n, ball_list):
    total = q ** n
    covers = [[] for _ in range(total)]
    for c in range(total):
        for w in ball_list[c]:
            covers[w].append(c)
    return covers


def _plain_greedy(q, n, ball_bits, ball_list, rng):
    total = q ** n
    uncovered_bits = (1 << total) - 1
    uncovered_count = total
    selected = []
    # Precompute ball_bits as a list of ints; gains via bit_count.
    # To speed up, we cache gains and only recompute when uncovered changes (always).
    while uncovered_count > 0:
        # vectorized-ish: compute gains for all centers
        # Use a list comprehension (C-level bit_count is fast)
        gains = [ (b & uncovered_bits).bit_count() for b in ball_bits ]
        maxg = max(gains)
        if maxg <= 0:
            break
        cands = [c for c in range(total) if gains[c] == maxg]
        best_c = rng.choice(cands)
        selected.append(best_c)
        uncovered_bits &= ~ball_bits[best_c]
        uncovered_count -= maxg
    return selected


def _criticality_greedy(q, n, ball_bits, ball_list, covers, rng):
    total = q ** n
    uncovered_bits = (1 << total) - 1
    uncovered_count = total
    selected = []
    crit = [len(covers[w]) for w in range(total)]
    while uncovered_count > 0:
        # find least-critical uncovered word
        best_w = -1
        best_crit = 10 ** 9
        ub = uncovered_bits
        while ub:
            lsb = ub & (-ub)
            w = lsb.bit_length() - 1
            ub ^= lsb
            cw = crit[w]
            if cw < best_crit:
                best_crit = cw
                best_w = w
                if best_crit <= 1:
                    break
        if best_w < 0:
            break
        cands = list(covers[best_w])
        rng.shuffle(cands)
        best_c = -1
        best_gain = -1
        for c in cands:
            gain = (ball_bits[c] & uncovered_bits).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c < 0 or best_gain <= 0:
            break
        selected.append(best_c)
        uncovered_bits &= ~ball_bits[best_c]
        uncovered_count -= best_gain
    return selected


def _local_search2(q, n, ball_bits, ball_list, covers, selected, rng, deadline):
    total = q ** n
    code = list(selected)
    code_set = set(code)

    def rebuild_cc():
        cc = np.zeros(total, dtype=np.int64)
        for c in code:
            cc[ball_list[c]] += 1
        return cc

    cc = rebuild_cc()

    # Phase 1: deletion sweeps
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(code)):
            c = code[i]
            bl = ball_list[c]
            if np.any(cc[bl] == 1):
                continue
            cc[bl] -= 1
            code.pop(i)
            code_set.discard(c)
            improved = True
            break

    # Phase 2: 1-for-1 swaps that enable a deletion
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        cc = rebuild_cc()
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            bl = ball_list[c]
            unique = bl[cc[bl] == 1]
            if len(unique) == 0:
                continue
            cand = set()
            for w in unique:
                for r in covers[w]:
                    if r not in code_set:
                        cand.add(r)
            if not cand:
                continue
            cand_list = list(cand)
            rng.shuffle(cand_list)
            for r in cand_list:
                if time.time() > deadline:
                    break
                new_cc = cc.copy()
                new_cc[ball_list[c]] -= 1
                new_cc[ball_list[r]] += 1
                if np.any(new_cc < 0):
                    continue
                # deletion sweep on candidate
                d_code = code[:i] + code[i+1:] + [r]
                d_cc = new_cc
                did = True
                while did:
                    if time.time() > deadline:
                        break
                    did = False
                    for j in range(len(d_code)):
                        dc = d_code[j]
                        dbl = ball_list[dc]
                        if np.any(d_cc[dbl] == 1):
                            continue
                        d_cc[dbl] -= 1
                        d_code.pop(j)
                        did = True
                        break
                if len(d_code) < len(code):
                    code = d_code
                    code_set = set(d_code)
                    cc = d_cc
                    improved = True
                    break
            if improved:
                break
    return code


def _perturb(code, total, rng, amount):
    code = list(code)
    for _ in range(amount):
        i = rng.randrange(len(code))
        code[i] = rng.randrange(total)
    seen = set()
    out = []
    for c in code:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _is_valid(code, ball_bits, total):
    covered = 0
    for c in code:
        covered |= ball_bits[c]
    return covered == ((1 << total) - 1)


def _to_codewords(code, n, q):
    out = []
    for c in code:
        d = _digits_from_index(c, n, q)
        out.append(list(d))
    return out


def construct(q, n, radius):
    start_time = time.time()
    total = q ** n

    ball_bits, ball_list = _precompute(q, n, radius)
    covers = _precompute_covers(q, n, ball_list)

    rng = random.Random(12345)

    best_code = None
    best_size = 10 ** 9

    def try_code(code):
        nonlocal best_code, best_size
        if _is_valid(code, ball_bits, total) and len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            return True
        return False

    # Initial: plain greedy + local search
    init_deadline = start_time + 20.0
    code0 = _plain_greedy(q, n, ball_bits, ball_list, rng)
    code0 = _local_search2(q, n, ball_bits, ball_list, covers, code0, rng, init_deadline)
    try_code(code0)

    # Iterated restarts
    restart_budget = 7.0
    while time.time() < DEADLINE - 6.0:
        attempt_start = time.time()
        if rng.random() < 0.5:
            code = _criticality_greedy(q, n, ball_bits, ball_list, covers, rng)
        else:
            code = _plain_greedy(q, n, ball_bits, ball_list, rng)
        if best_code is not None and rng.random() < 0.4:
            code = _perturb(code, total, rng, amount=2)
        ls_deadline = min(attempt_start + restart_budget, DEADLINE - 6.0)
        code = _local_search2(q, n, ball_bits, ball_list, covers, code, rng, ls_deadline)
        try_code(code)
        if best_code is not None and rng.random() < 0.3:
            ls_deadline2 = min(time.time() + restart_budget, DEADLINE - 6.0)
            code2 = _local_search2(q, n, ball_bits, ball_list, covers,
                                   _perturb(best_code, total, rng, 1), rng, ls_deadline2)
            try_code(code2)

    if best_code is None:
        best_code = code0

    # Final deletion sweep
    cc = np.zeros(total, dtype=np.int64)
    for c in best_code:
        cc[ball_list[c]] += 1
    improved = True
    while improved and time.time() < DEADLINE - 3.0:
        improved = False
        for i in range(len(best_code)):
            c = best_code[i]
            bl = ball_list[c]
            if np.any(cc[bl] == 1):
                continue
            cc[bl] -= 1
            best_code.pop(i)
            improved = True
            break

    return _to_codewords(best_code, n, q)
