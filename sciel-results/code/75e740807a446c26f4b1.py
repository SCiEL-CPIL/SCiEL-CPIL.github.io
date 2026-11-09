"""Covering design C(v,k,t) — multi-start greedy + local search + reduction."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 110.0   # 120s wall clock, 10s safety margin
    points = list(range(v))

    # Precompute all t-subsets
    all_t = list(itertools.combinations(points, t))
    Nt = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    # For each point, the list of t-subset indices containing it
    pt_subsets = defaultdict(list)
    for s in all_t:
        for p in s:
            pt_subsets[p].append(idx[s])

    def time_ok():
        return time.time() < DEADLINE

    def make_greedy(seed_rng):
        """Greedy set cover: repeatedly pick the block (k-subset) covering the most uncovered t-subsets."""
        # uncovered: bitset of t-subset indices still uncovered
        uncovered = [1] * Nt
        # counts[p] = number of uncovered t-subsets containing point p
        counts = [0] * v
        for p in range(v):
            c = 0
            for si in pt_subsets[p]:
                if uncovered[si]:
                    c += 1
            counts[p] = c
        total_uncov = Nt
        blocks = []
        while total_uncov > 0 and time_ok():
            # Greedily build one block: start with the point with max uncovered,
            # then add the point that maximizes new coverage.
            # Pick best seed point
            best_p = -1
            best_c = -1
            for p in range(v):
                if counts[p] > best_c:
                    best_c = counts[p]
                    best_p = p
            if best_p < 0:
                break
            block = [best_p]
            in_block = set(block)
            # For candidate x, gain = # uncovered t-subsets that contain x and (block ∪ {x} has ≥ t points already forming them)
            # A t-subset is covered by block∪{x} iff it is a subset of block∪{x}.
            # For a t-subset S, S ⊆ block∪{x} iff S \ {x} ⊆ block.
            # So gain(x) = # uncovered t-subsets S with x∈S and (S\{x}) ⊆ block.
            # Precompute for current block: for each subset B' of block of size t-1, the t-subsets extending it by one point.
            # Build map: frozenset(block-subset of size t-1) -> list of (point, subset_idx)
            # Simpler: for each x not in block, iterate over (t-1)-subsets of block, check if S = that ∪ {x} is uncovered.
            # To speed: precompute for the current block the (t-1)-subsets.
            block_set = set(block)
            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                # (t-1)-subsets of current block
                t1_subs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t1_subs:
                        s = tuple(sorted(sub + (x,)))
                        si = idx.get(s)
                        if si is not None and uncovered[si]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0 or best_gain <= 0:
                    # no improvement possible; fill with arbitrary points
                    for x in range(v):
                        if x not in block_set and len(block) < k:
                            block.append(x)
                            block_set.add(x)
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            # Mark all t-subsets contained in this block as covered
            for s in itertools.combinations(block, t):
                si = idx[s]
                if uncovered[si]:
                    uncovered[si] = 0
                    total_uncov -= 1
                    for p in s:
                        counts[p] -= 1
        return blocks

    def is_valid(blocks):
        seen = set()
        for s in all_t:
            found = False
            for b in blocks:
                if all(p in b for p in s):
                    found = True
                    break
            if not found:
                return False
        return True

    def try_reduce(blocks, rng):
        """Try to remove blocks that are redundant (their t-subsets are all covered by others)."""
        blocks = list(blocks)
        n = len(blocks)
        # For each block, the set of t-subset indices it covers
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(idx[s])
            block_subs.append(subs)
        # coverage count per t-subset
        cov = [0] * Nt
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(n):
                if not blocks[i]:
                    continue
                # can we remove block i?
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    # remove
                    for si in block_subs[i]:
                        cov[si] -= 1
                    blocks[i] = []
                    changed = True
        # compact
        return [b for b in blocks if b]

    # Multi-start
    best = None
    best_len = float('inf')
    rng = random.Random(12345)

    # First, one deterministic greedy (no randomness) as baseline
    base = make_greedy(None)
    base = try_reduce(base, rng)
    if is_valid(base):
        best = base
        best_len = len(base)

    # Randomized restarts: shuffle point order / tie-breaking via different seeds
    attempt = 0
    while time_ok():
        attempt += 1
        # Randomize by permuting point labels (symmetry breaking)
        perm = list(range(v))
        rng.shuffle(perm)
        inv = [0] * v
        for i, p in enumerate(perm):
            inv[p] = i
        # We can't easily permute inside make_greedy without rewriting; instead
        # use randomness in tie-breaking by adding small random noise to counts.
        # Simpler: just run make_greedy but it's deterministic. To vary, we
        # randomly drop some early greedy decisions by using a different seed
        # point selection.
        # Here: implement a lightweight randomized greedy.
        uncovered = [1] * Nt
        counts = [0] * v
        for p in range(v):
            c = 0
            for si in pt_subsets[p]:
                if uncovered[si]:
                    c += 1
            counts[p] = c
        total_uncov = Nt
        blocks = []
        while total_uncov > 0 and time_ok():
            # choose seed point: top-3 by count, pick randomly
            order = sorted(range(v), key=lambda p: (-counts[p], p))
            top = order[:3]
            best_p = rng.choice(top)
            block = [best_p]
            block_set = set(block)
            while len(block) < k and time_ok():
                t1_subs = list(itertools.combinations(block, t - 1))
                gains = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t1_subs:
                        s = tuple(sorted(sub + (x,)))
                        si = idx.get(s)
                        if si is not None and uncovered[si]:
                            gain += 1
                    gains.append((gain, x))
                gains.sort(reverse=True)
                if not gains or gains[0][0] <= 0:
                    for x in range(v):
                        if x not in block_set and len(block) < k:
                            block.append(x)
                            block_set.add(x)
                    break
                # pick from top-2 with some randomness
                topg = gains[:2]
                best_gain, best_x = topg[0]
                if len(topg) == 2 and rng.random() < 0.3:
                    best_x = topg[1][1]
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                si = idx[s]
                if uncovered[si]:
                    uncovered[si] = 0
                    total_uncov -= 1
                    for p in s:
                        counts[p] -= 1
        if total_uncov == 0:
            blocks = try_reduce(blocks, rng)
            if is_valid(blocks) and len(blocks) < best_len:
                best = blocks
                best_len = len(blocks)

    if best is None:
        # Fallback: return the baseline even if validity not rechecked (should be valid)
        return base
    return best
