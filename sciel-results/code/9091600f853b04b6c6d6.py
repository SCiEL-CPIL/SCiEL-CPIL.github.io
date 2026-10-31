"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

Structural redesign: separate block *generation* (fast bitset greedy) from block
*selection* (greedy set cover + block-exchange local search). A pre-generated
candidate pool lets local search trade blocks, something pure greedy cannot do.
"""
import time
import random
import itertools

_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    cov = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    _CACHE[key] = (t_subs, index, n, cov)
    return _CACHE[key]


def _gen_block(uncovered, v, k, t, index, t_subs, rng):
    """Build one block covering many currently-uncovered t-subsets."""
    # anchor on smallest-index uncovered t-subset
    anchor = None
    for sub in t_subs:
        if (uncovered >> index[sub]) & 1:
            anchor = sub
            break
    if anchor is None:
        return None
    block = set(anchor)
    while len(block) < k:
        best_x = None
        best_gain = -1.0
        sorted_block = sorted(block)
        for x in range(v):
            if x in block:
                continue
            gain = 0
            for rest in itertools.combinations(sorted_block, t - 1):
                m = 1 << index[tuple(sorted(rest + (x,)))]
                if (uncovered >> index[tuple(sorted(rest + (x,)))] & 1):
                    gain += 1
            gain += rng.random() * 1e-9
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None or best_gain <= 1e-9:
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.add(x)
            break
        block.add(best_x)
    return tuple(sorted(block))


def _greedy_select(cov_list, n, deadline):
    uncovered = (1 << n) - 1
    remaining = list(range(len(cov_list)))
    selected = []
    while uncovered and remaining:
        best_i = -1
        best_gain = 0
        for i in remaining:
            g = (cov_list[i] & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        uncovered &= ~cov_list[best_i]
        remaining.remove(best_i)
        if time.time() > deadline:
            break
    return selected, uncovered


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=6000):
    nblocks = len(cov_list)
    sel = list(selected)

    def build_count(sel):
        c = [0] * n
        for i in sel:
            m = cov_list[i]
            while m:
                lb = m & (-m)
                c[lb.bit_length() - 1] += 1
                m &= m - 1
        return c

    count = build_count(sel)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # drop pass
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            ok = True
            while m:
                lb = m & (-m)
                if count[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                m &= m - 1
            if ok:
                m2 = cov_list[i]
                while m2:
                    lb = m2 & (-m2)
                    count[lb.bit_length() - 1] -= 1
                    m2 &= m2 - 1
                sel.pop(pos)
                improved = True
                break
        if improved:
            continue
        # swap pass: drop i, add j covering the criticals -> may enable a drop
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            crit = []
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    crit.append(idx)
                m &= m - 1
            if not crit:
                continue
            for j in range(nblocks):
                if j in sel:
                    continue
                mj = cov_list[j]
                covers = True
                for idx in crit:
                    if not (mj >> idx) & 1:
                        covers = False
                        break
                if covers:
                    m2 = cov_list[i]
                    while m2:
                        lb = m2 & (-m2)
                        count[lb.bit_length() - 1] -= 1
                        m2 &= m2 - 1
                    m3 = cov_list[j]
                    while m3:
                        lb = m3 & (-m3)
                        count[lb.bit_length() - 1] += 1
                        m3 &= m3 - 1
                    sel[pos] = j
                    improved = True
                    break
            if improved:
                break
    return sel


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    rng = random.Random(12345)

    t_subs, index, n, cov = _prepare(v, k, t)

    # ---- Phase 1: generate diverse candidate pool ----
    candidates = []
    cand_set = set()
    full = (1 << n) - 1
    uncovered = full
    gens = 0
    gen_limit = 6000
    while uncovered and gens < gen_limit and time.time() < deadline - 15:
        b = _gen_block(uncovered, v, k, t, index, t_subs, rng)
        if b is None:
            break
        if b not in cand_set:
            cand_set.add(b)
            candidates.append(b)
        uncovered &= ~cov[b]
        gens += 1
        if gens % 40 == 0 and time.time() < deadline - 25:
            for _ in range(4):
                rb = tuple(sorted(rng.sample(range(v), k)))
                if rb not in cand_set:
                    cand_set.add(rb)
                    candidates.append(rb)

    # safety: ensure candidates can cover everything
    while uncovered and time.time() < deadline - 8:
        b = _gen_block(uncovered, v, k, t, index, t_subs, rng)
        if b is None:
            break
        if b not in cand_set:
            cand_set.add(b)
            candidates.append(b)
        uncovered &= ~cov[b]

    cov_list = [cov[b] for b in candidates]

    # ---- Phase 2: greedy selection ----
    selected, unc = _greedy_select(cov_list, n, deadline - 20)

    # ---- Phase 3: local search ----
    selected = _local_search(selected, cov_list, n, deadline - 5, rng)

    # ---- Fallback: guarantee completeness ----
    if time.time() < deadline - 2:
        uncovered_now = full
        for i in selected:
            uncovered_now &= ~cov_list[i]
        if uncovered_now:
            for j, m in enumerate(cov_list):
                if m & uncovered_now:
                    if j not in selected:
                        selected.append(j)
                    uncovered_now &= ~m
                    if not uncovered_now:
                        break

    return [list(candidates[i]) for i in selected]
