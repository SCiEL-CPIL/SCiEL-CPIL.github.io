# EVOLVE-BLOCK-START
import time
import math
import itertools
from collections import Counter

def construct(v, k, t):
    deadline = time.time() + 112.0
    all_tsubs = list(itertools.combinations(range(v), t))
    N = len(all_tsubs)
    idx = {c: i for i, c in enumerate(all_tsubs)}

    covered = bytearray(N)
    # dual weights: uncovered subsets start at 1.0
    w = [1.0] * N

    # index: point -> list of t-subset indices containing it
    point_subs = [[] for _ in range(v)]
    for i, c in enumerate(all_tsubs):
        for p in c:
            point_subs[p].append(i)

    # for each (v-1)-subset, precompute which t-subset indices it contains
    v1 = list(itertools.combinations(range(v), v - 1))
    v1_set = {c: i for i, c in enumerate(v1)}
    v1_contained = []
    for c in v1:
        cs = set(c)
        lst = []
        for i, tc in enumerate(all_tsubs):
            if all(p in cs for p in tc):
                lst.append(i)
        v1_contained.append(lst)

    blocks = []
    seen = set()

    def _price(S, wsum_S, max_bound, deadline):
        # branch-and-bound over S to find k-subset maximizing covered-weight
        S = list(S)
        nS = len(S)
        # precompute for each point: its weight contribution within S
        pt_w = [0.0] * nS
        for a in range(nS):
            pa = S[a]
            for j in point_subs[pa]:
                if not covered[j]:
                    # check other t-1 points in S
                    ok = True
                    for b in range(nS):
                        if b == a:
                            continue
                        if S[b] not in all_tsubs[j]:
                            ok = False
                            break
                    if ok:
                        pt_w[a] += w[j]

        best_w = 0.0
        best_mask = 0
        # bound: sum of top (k-|cur|) point weights (overestimates shared t-subsets)
        def rec(pos, chosen, cur_mask, cur_w):
            nonlocal best_w, best_mask
            if time.time() > deadline:
                return
            need = k - len(chosen)
            if need == 0:
                if cur_w > best_w:
                    best_w = cur_w
                    best_mask = cur_mask
                return
            if pos >= nS:
                return
            # upper bound using top point weights from pos..nS-1
            rem = pt_w[pos:]
            if len(rem) < need:
                return
            top = sorted(rem, reverse=True)[:need]
            ub = cur_w + sum(top)
            if ub <= best_w:
                return
            # branch: include pos
            chosen.append(pos)
            rec(pos + 1, chosen, cur_mask | (1 << pos), cur_w + pt_w[pos])
            chosen.pop()
            # branch: exclude pos
            rec(pos + 1, chosen, cur_mask, cur_w)

        rec(0, [], 0, 0.0)
        return best_mask, best_w

    def _add_block(subset):
        nonlocal w
        block = sorted(subset)
        if block in seen:
            return False
        seen.add(block)
        blocks.append(block)
        # update coverage and weights (multiplicative)
        for tc in itertools.combinations(block, t):
            j = idx[tc]
            if not covered[j]:
                covered[j] = 1
                w[j] *= 0.5
        return True

    # seed with the all-points block
    _add_block(range(v))

    while time.time() < deadline:
        # find best (v-1)-subset by covered-weight
        best_v1_w = -1.0
        best_v1 = None
        for i, c in enumerate(v1):
            s = 0.0
            for j in v1_contained[i]:
                if not covered[j]:
                    s += w[j]
            if s > best_v1_w:
                best_v1_w = s
                best_v1 = c
        if best_v1 is None or best_v1_w <= 0:
            break
        # price: choose k-subset of best_v1 maximizing weight
        mask, val = _price(best_v1, best_v1_w, best_v1_w + 1e-9, deadline)
        if val <= 0:
            break
        subset = [best_v1[a] for a in range(len(best_v1)) if mask & (1 << a)]
        if len(subset) < k:
            break
        _add_block(subset)

    # check coverage; if incomplete, fill with greedy
    if any(not covered[j] for j in range(N)) and time.time() < deadline:
        uncovered = [i for i in range(N) if not covered[i]]
        while uncovered and time.time() < deadline:
            # pick first uncovered t-subset as seed
            seed = all_tsubs[uncovered[0]]
            block = list(seed)
            while len(block) < k:
                best_p = None
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tc = tuple(sorted(rest + (p,)))
                        j = idx[tc]
                        if not covered[j]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                block.append(best_p)
            block.sort()
            _add_block(block)
            # refresh uncovered
            uncovered = [i for i in range(N) if not covered[i]]

    return blocks
# EVOLVE-BLOCK-END
