import itertools
import random
import time
from math import comb


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _block_t_mask(block, t):
    return frozenset(itertools.combinations(block, t))


def _bit_index_of_tuple(combo, index):
    # index maps a t-subset (ascending tuple) to a unique bit position
    return index[tuple(combo)]


def _greedy_build(v, k, t, index, deadline, rng):
    """Greedy set cover: repeatedly open a block on the most uncovered t-subset,
    then add points by a (greedy + random) score to reach k points."""
    n_sub = len(index)
    # coverage count per t-subset
    cov = [0] * n_sub
    blocks = []

    # Precompute, per point, the list of t-subsets (indices) that contain it.
    # A t-subset contains point x iff x is one of its members.
    subs_by_point = [[] for _ in range(v)]
    for sub, idx in index.items():
        for x in sub:
            subs_by_point[x].append(idx)

    while True:
        now = time.time()
        if now > deadline:
            break
        # find an uncovered t-subset to seed the block
        uncovered = [i for i in range(n_sub) if cov[i] == 0]
        if not uncovered:
            break
        # seed with the uncovered subset (any; use first deterministic-ish choice)
        seed = uncovered[0]
        block = set()
        # add points to maximize newly covered subsets
        for _ in range(k):
            now = time.time()
            if now > deadline:
                break
            best_x = None
            best_gain = -1
            best_random = -1.0
            for x in range(v):
                if x in block:
                    continue
                # count uncovered t-subsets that contain x AND can be completed
                # with the current block (i.e., share t-1 with block? not required for counting gain
                # since we only care whether adding x covers new subsets)
                # A t-subset is newly covered by adding x iff it currently uncovered AND
                # all other t-1 members are in block+{x}? No: adding x covers subset S iff
                # S contains x and S \ {x} subset block. So S must be subset of block U {x}
                # and contain x.
                gain = 0
                for idx in subs_by_point[x]:
                    if cov[idx] == 0:
                        # check S \ {x} subset block
                        S = index_inv[idx]  # will set below
                        ok = True
                        for m in S:
                            if m != x and m not in block:
                                ok = False
                                break
                        if ok:
                            gain += 1
                r = rng.random()
                if gain > best_gain or (gain == best_gain and r > best_random):
                    best_gain = gain
                    best_random = r
                    best_x = x
            if best_x is None:
                break
            block.add(best_x)
        block = sorted(block)
        # apply coverage
        bm = _block_t_mask(block, t)
        for S in bm:
            i = index[S]
            cov[i] += 1
        blocks.append(block)

    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (150s hard kill)
    rng = random.Random(123456789)

    # Build index of all t-subsets
    all_subs = _all_t_subsets(v, t)
    index = {}
    index_inv = {}
    for i, S in enumerate(all_subs):
        index[S] = i
        index_inv[i] = S

    n_sub = len(all_subs)

    # coverage count
    cov = [0] * n_sub
    blocks = []

    # per-point list of t-subset indices containing that point (for fast gain)
    subs_by_point = [[] for _ in range(v)]
    for S, i in index.items():
        for x in S:
            subs_by_point[x].append(i)

    # To test "S \ {x} subset block", precompute for each (point, subset) the set of other members
    others = {}
    for i, S in index_inv.items():
        others[i] = tuple(m for m in S)  # we'll filter x on the fly

    while time.time() < deadline:
        # find first uncovered
        uncovered = None
        for i in range(n_sub):
            if cov[i] == 0:
                uncovered = i
                break
        if uncovered is None:
            break

        # start block from this subset
        seed_S = index_inv[uncovered]
        block = set(seed_S)

        # add points to reach k
        while len(block) < k and time.time() < deadline:
            best_x = None
            best_gain = -1
            best_r = -1.0
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # iterate subsets containing x
                for i in subs_by_point[x]:
                    if cov[i] != 0:
                        continue
                    S = index_inv[i]
                    # need S \ {x} subset block
                    ok = True
                    for m in S:
                        if m != x and m not in block:
                            ok = False
                            break
                    if ok:
                        gain += 1
                r = rng.random()
                if gain > best_gain or (gain == best_gain and r > best_r):
                    best_gain = gain
                    best_r = r
                    best_x = x
            if best_x is None:
                break
            block.add(best_x)

        block = sorted(block)
        # apply
        for S in itertools.combinations(block, t):
            cov[index[S]] += 1
        blocks.append(block)

    # Polish: try to remove blocks that don't uniquely cover any t-subset
    if time.time() < deadline - 5:
        # coverage count again
        cov2 = [0] * n_sub
        for b in blocks:
            for S in itertools.combinations(b, t):
                cov2[index[S]] += 1
        removed = True
        while removed and time.time() < deadline - 3:
            removed = False
            for bi in range(len(blocks) - 1, -1, -1):
                b = blocks[bi]
                # check if any t-subset is covered only by this block
                unique = False
                for S in itertools.combinations(b, t):
                    if cov2[index[S]] == 1:
                        unique = True
                        break
                if not unique:
                    for S in itertools.combinations(b, t):
                        cov2[index[S]] -= 1
                    blocks.pop(bi)
                    removed = True

    return blocks
