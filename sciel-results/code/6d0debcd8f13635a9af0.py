"""Covering design C(v,k,t): greedy construction + local-search refinement."""
# EVOLVE-BLOCK-START
import itertools
import time


def _insert_sorted(arr, x):
    """Return a new tuple with x inserted into the sorted tuple arr (x not in arr)."""
    n = len(arr)
    i = 0
    while i < n and arr[i] < x:
        i += 1
    return arr[:i] + (x,) + arr[i:]


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before the 150s kill
    points = list(range(v))
    all_t = list(itertools.combinations(range(v), t))
    uncovered = set(all_t)

    def compute_gain(block, x):
        """Number of currently-uncovered t-subsets covered by adding x to block."""
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            cand = _insert_sorted(rest, x)
            if cand in uncovered:
                gain += 1
        return gain

    def greedy_block(anchor):
        block = list(anchor)
        block_set = set(block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in points:
                if x in block_set:
                    continue
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                for x in points:
                    if x not in block_set:
                        best_x = x
                        break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        return block

    # ---- Phase 1: greedy construction ----
    blocks = []
    while uncovered and time.time() < deadline - 15.0:
        anchor = min(uncovered)
        block = greedy_block(anchor)
        blocks.append(block)
        for comb in itertools.combinations(block, t):
            uncovered.discard(comb)

    # Guarantee completeness (no deadline pressure on the remainder).
    while uncovered:
        anchor = min(uncovered)
        block = greedy_block(anchor)
        if block not in blocks:
            blocks.append(block)
        for comb in itertools.combinations(block, t):
            uncovered.discard(comb)

    # ---- Phase 2: local search -- drop redundant blocks ----
    # For each t-subset, track the number of blocks covering it.
    cover_count = {}
    for block in blocks:
        for comb in itertools.combinations(block, t):
            cover_count[comb] = cover_count.get(comb, 0) + 1

    n = len(blocks)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(n):
            if time.time() > deadline:
                break
            # A block is redundant iff every t-subset it covers is covered >= 2 times.
            redundant = True
            for comb in itertools.combinations(blocks[i], t):
                if cover_count[comb] <= 1:
                    redundant = False
                    break
            if redundant:
                for comb in itertools.combinations(blocks[i], t):
                    cover_count[comb] -= 1
                blocks.pop(i)
                n -= 1
                changed = True
                break  # indices shifted; restart the scan

    return blocks
# EVOLVE-BLOCK-END
