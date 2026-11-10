"""Covering design C(v, k, t) – randomized greedy with deadline guard."""
import itertools
import time
import random

_DEADLINE_S = 112.0  # Safety margin from 120s evaluator limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _precompute_combo_tables(v, k, t):
    """Precompute (a) all (t-1)-subsets of each (t-1)-subset for gain calc,
    and (b) all t-subsets of each possible block (as frozenset keys) for coverage
    marking.  We cache per (v,k,t)."""
    key = (v, k, t)
    if key in _COMBO_CACHE:
        return _COMBO_CACHE[key]

    # For gain: for each (t-1)-tuple (sorted) we need the list of (t-1)-tuples
    # Actually the greedy loop iterates over combinations(block, t-1) dynamically.
    # Instead, precompute for each possible (t-1)-subset S (as a frozenset),
    # the list of t-subsets that contain S (i.e., S ∪ {x} for x not in S).
    # But that's large.  Simpler: precompute all (t-1)-subsets of the v points
    # and for each, the list of t-subsets it extends to.
    #
    # Alternative: precompute for each point x, the list of (t-1)-subsets not
    # containing x.  Then gain(x) = |{S in combos(block, t-1) : S ∪ {x} in uncovered}|.
    #
    # Let's precompute:
    #   t1_subs: list of all (t-1)-tuples (sorted)
    #   For each t-1-subset S (as a set), the list of t-subsets S∪{x}
    t1_list = list(itertools.combinations(range(v), t - 1))
    # Map from frozenset -> list of t-subset tuples
    t1_to_t = {}
    for s in t1_list:
        sset = frozenset(s)
        # x not in s
        lst = []
        for x in range(v):
            if x not in sset:
                lst.append(tuple(sorted(s + (x,))))
        t1_to_t[sset] = lst

    # Precompute all t-subsets of each possible block?  Too many blocks.
    # Instead, for marking coverage after adding a block, we just iterate
    # itertools.combinations(block, t).  That's fine.

    _COMBO_CACHE[key] = (t1_list, t1_to_t)
    return _COMBO_CACHE[key]


_COMBO_CACHE = {}


def _greedy_block(uncovered, v, k, t, seed, t1_to_t):
    """Build a k-subset containing seed, maximizing coverage of uncovered t-subsets."""
    block = list(seed)
    block_set = set(block)
    # Maintain the set of (t-1)-subsets of the current block for fast gain
    # We'll recompute combos(block, t-1) each step; for k<=7, t<=4, this is tiny.
    while len(block) < k and _time_left() > 0.3:
        best_pt = None
        best_gain = -1
        # Precompute the (t-1)-subsets of the current block
        block_t1 = list(itertools.combinations(block, t - 1))
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for combo in block_t1:
                sub = tuple(sorted(combo + (x,)))
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = x
        if best_pt is None:
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            best_pt = random.choice(candidates)
        block.append(best_pt)
        block_set.add(best_pt)
    # Fill if needed
    if len(block) < k:
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block_set:
                block.append(x)
                block_set.add(x)
    return sorted(block)


def _remove_redundant_blocks(blocks, v, k, t, deadline_frac=0.15):
    """Remove blocks that are not needed to cover all t-subsets.
    Uses a greedy 'essential' check: a block is removable if every t-subset
    it covers is also covered by some other block.  We do this in a loop
    until no more removals.  Time-bounded."""
    time_budget = _DEADLINE_S * deadline_frac
    t0 = time.time()
    # Count how many blocks cover each t-subset
    sub_counts = {}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            sub_counts[sub] = sub_counts.get(sub, 0) + 1

    n = len(blocks)
    changed = True
    while changed and (time.time() - t0) < time_budget:
        changed = False
        for i in range(n):
            if i >= len(blocks):
                break
            b = blocks[i]
            # Check if any t-subset of b is covered only by this block
            removable = True
            for sub in itertools.combinations(b, t):
                if sub_counts.get(sub, 0) <= 1:
                    removable = False
                    break
            if removable:
                # Decrement counts
                for sub in itertools.combinations(b, t):
                    sub_counts[sub] -= 1
                blocks.pop(i)
                n -= 1
                changed = True
                break  # restart scan since indices shifted
    return blocks


def construct(v, k, t):
    random.seed(12345)
    all_tsubs = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_len = float('inf')

    # Precompute combo tables for gain computation
    t1_list, t1_to_t = _precompute_combo_tables(v, k, t)

    # Multiple randomized restarts
    max_restarts = 30
    for r in range(max_restarts):
        if _time_left() < 2.0:
            break
        uncovered = set(all_tsubs)
        blocks = []
        block_set = set()

        while uncovered and _time_left() > 1.0:
            # Pick target: prefer uncovered subsets with high "constraint"
            # (few points that appear in already-chosen blocks? Just use random for diversity)
            if len(uncovered) > 200:
                sample = random.sample(list(uncovered), min(100, len(uncovered)))
                target = random.choice(sample)
            else:
                target = random.choice(list(uncovered))

            block = _greedy_block(uncovered, v, k, t, seed=target, t1_to_t=t1_to_t)
            bkey = tuple(block)
            if bkey not in block_set:
                blocks.append(block)
                block_set.add(bkey)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)

        if not uncovered:
            # Try to reduce block count via redundant removal
            reduced = _remove_redundant_blocks(blocks, v, k, t, deadline_frac=0.10)
            if len(reduced) < best_len:
                best_blocks = reduced
                best_len = len(reduced)
                if best_len <= 151:  # Known best
                    break

    # Fallback if no valid covering found in time
    if best_blocks is None:
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = _greedy_block(uncovered, v, k, t, seed=target, t1_to_t=t1_to_t)
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        best_blocks = blocks

    return best_blocks
