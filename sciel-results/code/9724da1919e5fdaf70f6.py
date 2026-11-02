"""Improved greedy covering with dynamic heap-based point selection."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Represent t-subsets as frozensets for fast membership
    # We track uncovered t-subsets in a set
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    # For efficiency, precompute all (t-1)-subsets for each potential block extension
    # We'll use a heap to select the best point to add to the current block

    while uncovered:
        if time.time() > deadline:
            break

        # Start with the lexicographically smallest uncovered t-subset
        seed = min(uncovered)
        block = list(seed)

        # Build a set of the current block for fast lookup
        block_set = set(block)

        # We'll greedily add points that maximize the number of newly covered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate point x not in block, compute how many uncovered
            # t-subsets would be covered by adding x to the block
            # A t-subset is covered if it's a subset of the new block
            # i.e., it contains x and (t-1) elements from the current block

            # Precompute all (t-1)-subsets of the current block
            t_minus_1_subs = list(itertools.combinations(block, t - 1))

            # Count for each candidate point
            best_gain = -1
            best_x = -1

            # Use a smarter approach: for each uncovered t-subset, check if it can
            # be completed by adding one point from the current block
            # Actually, let's just iterate candidates and count

            # For performance, we can count by iterating over uncovered subsets
            # that contain exactly (t-1) elements from the block and 1 element outside

            # Build a mapping: for each (t-1)-subset of block, which uncovered t-subsets
            # extend it with one more point?
            # This is expensive. Let's just do direct counting for each candidate.

            candidates = [x for x in range(v) if x not in block_set]

            # To speed up, we can precompute for each uncovered t-subset, how many
            # of its elements are in the block
            # Then a candidate x covers an uncovered subset S if x in S and |S ∩ block| == t-1

            # Let's build a dictionary: for each point x, count of uncovered subsets
            # S where x in S and |S ∩ block| == t-1
            gains = {}
            for x in candidates:
                gains[x] = 0

            for sub in uncovered:
                # Count how many elements of sub are in block
                count_in_block = sum(1 for elem in sub if elem in block_set)
                if count_in_block == t - 1:
                    # The one element not in block is the candidate
                    for elem in sub:
                        if elem not in block_set:
                            gains[elem] = gains.get(elem, 0) + 1
                            break

            # Find the candidate with maximum gain
            best_gain = -1
            best_x = -1
            for x in candidates:
                g = gains.get(x, 0)
                if g > best_gain or (g == best_gain and x < best_x):
                    best_gain = g
                    best_x = x

            if best_x == -1:
                break

            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break

        block.sort()
        blocks.append(block)

        # Remove all t-subsets of this block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks
# EVOLVE-BLOCK-END
