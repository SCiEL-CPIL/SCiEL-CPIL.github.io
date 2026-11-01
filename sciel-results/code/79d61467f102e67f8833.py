# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter


def _comb_index(c, t):
    """Combinatorial number system: rank of sorted t-subset c in 0..C(n,t)-1."""
    n = c[-1] + 1
    idx = 0
    for i in range(t):
        idx += 1
        for j in range(c[i], n):
            idx -= 1
    return idx


def _gen_blocks(v, k, t):
    """Yield candidate k-subsets; random order, then deterministic (for restarts)."""
    pts = list(range(v))
    random.shuffle(pts)
    for comb in itertools.combinations(pts, k):
        yield comb
    for comb in itertools.combinations(range(v), k):
        yield comb


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 135.0          # safety margin below 150s kill
    rng = random.Random(20240517)

    # --- Build coverage index: t-set -> integer index ---
    all_tsets = list(itertools.combinations(range(v), t))
    total_t = len(all_tsets)
    idx_of = {}
    for i, c in enumerate(all_tsets):
        idx_of[c] = i

    def block_indices(block):
        return [_comb_index(tuple(sorted(c)), t) for c in itertools.combinations(block, t)]

    # ---------- Phase 1: block-level greedy set-cover construction ----------
    def build_greedy():
        cov = [0] * total_t
        blocks = []
        for b in _gen_blocks(v, k, t):
            if time.time() > deadline:
                break
            idxs = block_indices(b)
            # marginal coverage = number of still-uncovered t-sets in this block
            gain = 0
            for i in idxs:
                if cov[i] == 0:
                    gain += 1
            if gain > 0:
                blocks.append(frozenset(b))
                for i in idxs:
                    cov[i] += 1
                if all(c > 0 for c in cov):
                    break
        return blocks, cov

    # ---------- Phase 2: remove redundant blocks ----------
    def prune(blocks, cov):
        removed = True
        while removed:
            removed = False
            for bi in range(len(blocks)):
                b = blocks[bi]
                for i in block_indices(b):
                    if cov[i] == 1:
                        break
                else:
                    for i in block_indices(b):
                        cov[i] -= 1
                    del blocks[bi]
                    removed = True
                    break
        return blocks, cov

    # ---------- Phase 3: pairwise-merge local search ----------
    def merge_pass(blocks, cov, limit):
        n = len(blocks)
        i = 0
        while i < n and time.time() < limit:
            j = i + 1
            while j < n and time.time() < limit:
                A = blocks[i]; B = blocks[j]
                # t-sets uniquely covered by A or B
                uniqA = [x for x in block_indices(A) if cov[x] == 1]
                uniqB = [x for x in block_indices(B) if cov[x] == 1]
                if len(uniqA) + len(uniqB) <= k:
                    # candidate: union of the underlying points of those t-sets
                    cand = set()
                    for x in uniqA + uniqB:
                        cand |= set(all_tsets[x])
                    if len(cand) <= k:
                        # verify every t-set in cand is covered by A or B
                        ok = True
                        for c in itertools.combinations(sorted(cand), t):
                            ci = idx_of[c]
                            if cov[ci] < 1:
                                ok = False
                                break
                        if ok:
                            # apply merge
                            for x in block_indices(A):
                                cov[x] -= 1
                            for x in block_indices(B):
                                cov[x] -= 1
                            del blocks[i]; del blocks[j]
                            n = len(blocks)
                            newb = frozenset(cand)
                            for x in block_indices(newb):
                                cov[x] += 1
                            blocks.append(newb)
                            n = len(blocks)
                            i = 0
                            break
                j += 1
            i += 1
        return blocks, cov

    # ---------- Assemble with time budget ----------
    blocks, cov = build_greedy()
    if not all(c > 0 for c in cov):
        # incomplete: fall back to a plain greedy that is guaranteed to finish
        return _fallback_greedy(v, k, t)

    blocks, cov = prune(blocks, cov)

    # local search, multiple passes while time remains
    while time.time() < deadline - 5.0 and len(blocks) > 0:
        before = len(blocks)
        blocks, cov = merge_pass(blocks, cov, deadline - 5.0)
        blocks, cov = prune(blocks, cov)
        if len(blocks) >= before:
            break

    return [list(sorted(b)) for b in blocks]


def _fallback_greedy(v, k, t):
    """Simple per-point greedy guaranteeing completion (used only if phase 1 stalls)."""
    t0 = time.time()
    deadline = t0 + 135.0
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        block = list(min(uncovered))
        while len(block) < k:
            best = None; bestg = -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        g += 1
                if g > bestg:
                    bestg = g; best = x
            block.append(best)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
