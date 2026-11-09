"""Algebraic multi-scale (union of scaled geometric blocks) construction.

Different design family from the current program: instead of random sets +
add/remove local search, we build U as a union of k disjoint, scaled geometric
progressions.  Because the blocks are placed far apart (scale factor > block
span), the cross-block differences a-b (a in block i, b in block j, i!=j) are
ALL distinct, so |U-U| grows roughly like n^2 while |U+U| grows only ~ n^2/2
within-block plus n^2 cross terms that are mostly distinct -> a large ratio.
We then refine only the per-block scaling factors (coordinate ascent), which is
a far smaller, more structured search space than the current program's.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

# ---------- core computation (exact, small sets) ----------
def _set_sum_diff(U):
    Us = list(U)
    plus = set()
    for a in Us:
        for b in Us:
            plus.add(a + b)
    minus = set()
    for a in Us:
        for b in Us:
            minus.add(a - b)
    return len(plus), len(minus)

def _score(U):
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

# ---------- algebraic multi-scale construction ----------
def _build(k, bases, scales):
    """U = union over blocks i of { s_i * base_i^j : j=0..k-1 }, plus 0.
    Blocks are ordered by increasing scale so they are disjoint (scale > span)."""
    U = {0}
    for i in range(len(bases)):
        base = bases[i]
        s = scales[i]
        v = s
        for j in range(k):
            U.add(v)
            v *= base
    return sorted(U)

def _span(block_base, k):
    """max element of a geometric block with given base and length k (unit scale)."""
    return block_base ** (k - 1)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0   # safety margin before 150s kill
    rng = random.Random(777)

    best_val = -1e9
    best_set = [0, 1, 3]

    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # baseline
    consider([0, 1, 3])

    # ---------- coordinate-ascent over per-block scales ----------
    # A configuration = (k, bases, scales).  We fix k and bases, and optimise
    # the scales (which control block placement / max U) by coordinate ascent.
    def optimize_config(k, bases):
        """Given k (block length) and bases, optimise scales via coordinate ascent."""
        nblocks = len(bases)
        # initial scales: greedily place each block just beyond the previous span
        scales = []
        cur = 1
        for i in range(nblocks):
            scales.append(cur)
            cur = cur * _span(bases[i], k) + 1
        # ensure strictly increasing and disjoint
        for i in range(1, nblocks):
            if scales[i] <= scales[i - 1]:
                scales[i] = scales[i - 1] + 1

        cur_U = _build(k, bases, scales)
        cur_v, _ = _score(cur_U)
        best_local_v = cur_v
        best_local_scales = list(scales)

        improved = True
        it = 0
        while improved and it < 40 and time.time() < deadline - 3.0:
            improved = False
            it += 1
            for i in range(nblocks):
                # candidate new scales for block i
                base_span = _span(bases[i], k)
                cands = set()
                s0 = scales[i]
                # multiplicative neighbours
                for f in [0.5, 2.0, 3.0, 0.33]:
                    cands.add(max(1, int(round(s0 * f))))
                # just-beyond-previous-span placement
                if i > 0:
                    prev_end = scales[i - 1] * base_span
                    cands.add(prev_end + 1)
                    cands.add(prev_end + 2)
                # just-beyond-next-start to keep disjoint forward
                if i < nblocks - 1:
                    cands.add(max(1, scales[i + 1] // max(1, base_span) - 1))
                # a few random scales in a reasonable window
                lo = max(1, s0 // 4)
                hi = s0 * 4
                for _ in range(4):
                    cands.add(rng.randint(lo, hi))
                for c in sorted(cands):
                    if c < 1:
                        continue
                    # enforce disjointness: block i span must not overlap block i+1
                    # (simple check: ensure scales strictly increasing enough)
                    new_scales = list(scales)
                    new_scales[i] = c
                    # quick disjointness filter
                    ok = True
                    for j in range(nblocks - 1):
                        end_j = new_scales[j] * _span(bases[j], k)
                        start_next = new_scales[j + 1]
                        if start_next <= end_j:
                            ok = False
                            break
                    if not ok:
                        continue
                    U = _build(k, bases, new_scales)
                    v, s = _score(U)
                    if v > cur_v + 1e-12:
                        cur_v = v
                        scales = new_scales
                        cur_U = s
                        improved = True
                        if v > best_local_v:
                            best_local_v = v
                            best_local_scales = list(scales)
        consider(cur_U)
        return best_local_v, best_local_scales

    # ---------- search over (k, bases) space ----------
    # small k (block length) and few blocks; bases small to keep max U modest
    base_choices = [2, 3, 4, 5]
    k_choices = [2, 3, 4, 5]
    nblock_choices = [2, 3, 4]

    # deterministic sweep first
    for k in k_choices:
        for nb in nblock_choices:
            for bases in [tuple([b] * nb) for b in base_choices]:
                if time.time() > deadline - 5.0:
                    break
                optimize_config(k, bases)
            # mixed bases
            for bases in [(2, 3), (3, 2), (2, 3, 5), (5, 3, 2), (2, 4), (4, 2)][:nb]:
                if len(bases) != nb:
                    continue
                if time.time() > deadline - 5.0:
                    break
                optimize_config(k, bases)

    # ---------- randomised restarts over configurations ----------
    while time.time() < deadline - 5.0:
        k = rng.choice(k_choices)
        nb = rng.choice(nblock_choices)
        bases = tuple(rng.choice(base_choices) for _ in range(nb))
        optimize_config(k, bases)

    # ---------- final local polish on best set (add/remove) ----------
    if time.time() < deadline - 3.0:
        cur = list(best_set)
        mx = max(cur) if cur else 0
        improved = True
        it = 0
        while improved and time.time() < deadline - 2.0 and it < 80:
            improved = False
            it += 1
            cand = [1, 2, 3, 5, 8, 13, mx + 1, mx * 2 + 1, mx * 3 + 1,
                    mx + mx // 2, max(1, mx // 3)]
            cand = [c for c in cand if c >= 0 and c not in cur]
            rng.shuffle(cand)
            for c in cand:
                U = cur + [c]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break
            for c in list(cur):
                if c == 0:
                    continue
                U = [x for x in cur if x != c]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break

    return best_set
# EVOLVE-BLOCK-END
