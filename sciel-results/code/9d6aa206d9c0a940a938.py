# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

# Use a fixed seed for reproducibility
random.seed(42)


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a fast greedy algorithm
    with precomputed incidence structure, followed by local search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for the 120s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tset = set(all_tsubs)
    num_tsubs = len(all_tset)

    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tset)}

    # For each point, precompute the list of t-subsets containing that point
    point_to_tsubs = [[] for _ in range(v)]
    for ts in all_tset:
        for p in ts:
            point_to_tsubs[p].append(ts)

    # Convert to sets for O(1) membership tests
    point_to_tset = [set(lst) for lst in point_to_tsubs]

    # Greedy construction: maintain a list of which t-subsets are covered
    # covered[i] = number of blocks that cover t-subset i (or just use a boolean)
    # For speed, use a list of booleans
    covered = [False] * num_tsubs
    uncovered_count = num_tsubs

    blocks = []

    # Precompute for each pair of (partial_block, candidate_point) the gain
    # But this is too expensive. Instead, compute gain on the fly but efficiently.

    # For a partial block (list of points so far), the gain of adding point x
    # is the number of uncovered t-subsets that are subsets of partial_block + {x}
    # A t-subset S is covered by partial_block + {x} iff S ⊆ partial_block ∪ {x}
    # This means S contains x and S \ {x} ⊆ partial_block, OR S ⊆ partial_block (already covered)
    # So new coverage = t-subsets S where x ∈ S, S \ {x} ⊆ partial_block, and S not yet covered

    # For a partial block B of size m, the t-subsets newly covered by adding x
    # are those S ⊆ B ∪ {x} with x ∈ S and |S| = t and S not in covered
    # These are: choose (t-1) elements from B to form S = C(B, t-1) ∪ {x}
    # But only if t-1 <= m, i.e., m >= t-1

    # Let's use a different approach: for each block we build, compute all t-subsets
    # it covers and mark them.

    # For efficiency in the inner loop, let's precompute combinations of sizes up to t-1
    # for each possible partial block. But partial blocks vary.

    # Alternative: use the incidence vector approach
    # For each point x, precompute the bitset (or list) of t-subsets containing x
    # We already have point_to_tset[x]

    # For a partial block B, the t-subsets covered by B ∪ {x} that contain x
    # are: {S ∈ point_to_tset[x] : S \ {x} ⊆ B}
    # This is equivalent to: S ⊆ B ∪ {x} and x ∈ S

    # Let's define a helper to compute gain for a partial block and candidate x
    def compute_gain(partial_block, x, covered_arr):
        if x in partial_block:
            return 0
        # We need to count t-subsets S such that:
        # 1. x ∈ S
        # 2. S \ {x} ⊆ partial_block
        # 3. S is not yet covered
        # S \ {x} is a (t-1)-subset of partial_block
        # So we iterate over all (t-1)-subsets of partial_block
        gain = 0
        if len(partial_block) >= t - 1:
            for rest in itertools.combinations(partial_block, t - 1):
                ts = tuple(sorted(rest + (x,)))
                idx = tsub_index.get(ts)
                if idx is not None and not covered_arr[idx]:
                    gain += 1
        return gain

    # Greedy loop
    while uncovered_count > 0:
        if time.time() > deadline - 30:
            # Not enough time for another block; break and return what we have
            # But we need a valid covering. Let's just finish quickly if close.
            pass

        # Find the first uncovered t-subset to start a new block
        # For efficiency, find the uncovered t-subset with the fewest covering points
        # But for simplicity, just pick the first one
        start_ts = None
        for i in range(num_tsubs):
            if not covered[i]:
                start_ts = all_tset[i]
                break

        if start_ts is None:
            break

        # Build a block greedily
        block = list(start_ts)
        block_set = set(block)

        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                g = compute_gain(block, x, covered)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                # No point adds coverage; break or add arbitrary
                if best_x != -1:
                    block.append(best_x)
                    block_set.add(best_x)
                break
            block.append(best_x)
            block_set.add(best_x)

        # Sort the block for canonical form
        block.sort()

        # Add the block and update coverage
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None and not covered[idx]:
                covered[idx] = True
                uncovered_count -= 1

        if uncovered_count <= 0:
            break

    # At this point, we have a valid covering if uncovered_count == 0
    # But due to time limits, we might not have finished. Let's check.
    if uncovered_count > 0 and time.time() < deadline - 5:
        # Continue greedy with a simpler approach to ensure validity
        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            for i in range(num_tsubs):
                if not covered[i]:
                    # Add a block that covers this t-subset
                    ts = all_tset[i]
                    block = list(ts)
                    block_set = set(block)
                    while len(block) < k:
                        best_x = -1
                        best_gain = -1
                        for x in range(v):
                            if x in block_set:
                                continue
                            g = compute_gain(block, x, covered)
                            if g > best_gain:
                                best_gain = g
                                best_x = x
                        if best_x == -1:
                            # Pick an arbitrary unused point
                            for x in range(v):
                                if x not in block_set:
                                    best_x = x
                                    break
                        if best_x != -1:
                            block.append(best_x)
                            block_set.add(best_x)
                    block.sort()
                    # Avoid duplicates
                    if block not in blocks:
                        blocks.append(block)
                        for ts2 in itertools.combinations(block, t):
                            idx = tsub_index.get(ts2)
                            if idx is not None and not covered[idx]:
                                covered[idx] = True
                                uncovered_count -= 1
                    break

    # Local search: try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 20 and len(blocks) > 10:
        # Try to remove blocks one by one
        i = 0
        while i < len(blocks) and time.time() < deadline - 10:
            # Check if blocks[i] is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    # Check if any other block covers this t-subset
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if ts in set(itertools.combinations(blocks[j], t)):
                            redundant = False
                            break
                if not redundant:
                    break
            if redundant:
                blocks.pop(i)
            else:
                i += 1

    return blocks
# EVOLVE-BLOCK-END
