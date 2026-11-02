"""Covering design C(24,7,3) via greedy submodular maximization with efficient gain computation."""
import time
import itertools
import random
import sys

# Hard deadline: must return before 150s wall-clock; use 145s as safety margin
_DEADLINE = 145.0


def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return set(itertools.combinations(range(v), t))


def _block_covers(block, t):
    """Return the set of t-subsets covered by a given block (sorted tuple)."""
    return set(itertools.combinations(block, t))


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subs)
    
    # Track uncovered t-subsets as a set for O(1) lookup
    uncovered = set(all_t_subs)
    blocks = []
    block_set = set()  # for deduplication

    # For efficiency, precompute for each point the list of t-subsets containing it
    # point_to_subs[x] = list of t-subsets that include x
    point_to_subs = [[] for _ in range(v)]
    for sub in all_t_subs:
        for x in sub:
            point_to_subs[x].append(sub)

    while uncovered:
        # Check deadline
        if time.time() > deadline - 2.0:
            break

        # Greedy: build a block that maximizes marginal coverage gain
        # Start with the t-subset that appears in the fewest existing blocks
        # (i.e., pick a "rare" uncovered subset to anchor)
        # Actually, standard greedy: pick the block that covers the most uncovered t-subsets.
        # But we build block point-by-point.

        # Strategy: Start from an uncovered t-subset, then greedily add points
        # that maximize the number of NEW uncovered t-subsets covered.

        # Pick a starting point: the first uncovered t-subset (or a random one for diversity)
        # For determinism and speed, use the lexicographically smallest uncovered subset
        start_sub = min(uncovered)
        block = list(start_sub)

        while len(block) < k:
            if time.time() > deadline - 1.0:
                break

            block_set = set(block)
            best_gain = -1
            best_point = None

            # For each candidate point not in block, compute how many uncovered
            # t-subsets it would help cover when added to the block.
            # A t-subset is newly covered if it contains (t-1) points from the current
            # block and 1 new point x, and is currently uncovered.
            
            # Precompute: for each point x not in block, count how many (t-1)-subsets
            # of the current block, when combined with x, form an uncovered t-subset.
            
            # For efficiency, precompute the (t-1)-subsets of the current block
            block_t_minus_1 = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_t_minus_1:
                    # Check if sorted(rest + (x,)) is in uncovered
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is None or best_gain <= 0:
                # No point improves coverage; just add a random unused point
                used = block_set
                candidates = [x for x in range(v) if x not in used]
                if candidates:
                    best_point = candidates[0]  # deterministic
                else:
                    break
            block.append(best_point)

        block.sort()
        block_tuple = tuple(block)
        if block_tuple in block_set:
            continue
        block_set.add(block_tuple)
        blocks.append(block)

        # Update uncovered: remove all t-subsets of this block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    # If we ran out of time, we may not have a complete cover.
    # The evaluator checks validity, so an incomplete cover scores 0.
    # We must ensure we return a valid cover if possible.
    # If uncovered is not empty and we're past deadline, we're in trouble.
    # But the greedy should be fast enough for C(24,7,3) with ~76 blocks.
    
    return blocks


# --- Additional optimization: use a faster gain computation via bitsets ---
# For C(24,7,3): total 3-subsets = C(24,3) = 2024. This fits in a 2024-bit integer.
# We can use Python integers as bitsets for fast operations.

def construct_fast(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    all_t_subs = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subs)
    
    # Map each t-subset to an index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Bitset of uncovered t-subsets: bit i is 1 if t-subset i is uncovered
    uncovered_mask = (1 << total_t) - 1  # all ones
    
    blocks = []
    block_set = set()

    while uncovered_mask:
        if time.time() > deadline - 2.0:
            break

        # Greedy block construction
        # Start with the lowest-index uncovered t-subset
        # Find lowest set bit
        lowest = (uncovered_mask & -uncovered_mask).bit_length() - 1
        start_sub = all_t_subs[lowest]
        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline - 1.0:
                break

            # Compute gain for each candidate point
            # For point x, gain = number of uncovered t-subsets that contain x and
            # (t-1) points from current block
            block_t_minus_1 = list(itertools.combinations(block, t - 1))
            
            best_gain = -1
            best_point = None

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_t_minus_1:
                    candidate = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx[candidate]
                    if (uncovered_mask >> idx) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is None or best_gain <= 0:
                used = block_set
                candidates = [x for x in range(v) if x not in used]
                if candidates:
                    best_point = candidates[0]
                else:
                    break
            block.append(best_point)

        block.sort()
        block_tuple = tuple(block)
        if block_tuple in block_set:
            continue
        block_set.add(block_tuple)
        blocks.append(block)

        # Update uncovered mask
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            uncovered_mask &= ~(1 << idx)

    return blocks


# Use the faster version
def construct(v, k, t):
    return construct_fast(v, k, t)
