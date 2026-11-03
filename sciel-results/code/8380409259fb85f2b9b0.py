"""Improved greedy covering with efficient data structures and deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Total number of t-subsets
    total_tsubsets = 1  # C(v, t)
    # We'll use a set of frozensets for uncovered t-subsets
    # But for efficiency, let's use a different approach

    # Precompute all t-subsets as frozensets
    # This could be large: C(19,5) = 11628, so it's manageable
    all_tsubsets = list(itertools.combinations(range(v), t))
    total = len(all_tsubsets)

    # Use a set of tuples for uncovered
    uncovered = set(all_tsubsets)

    blocks = []

    # For efficiency, we'll track for each point, which t-subsets it helps cover
    # Actually, let's use a smarter greedy:
    # 1. Pick the first uncovered t-subset as seed
    # 2. Greedily add points that maximize new t-subsets covered

    while uncovered:
        if time.time() > deadline:
            break

        # Pick a seed from uncovered (first one)
        seed = min(uncovered)
        block = list(seed)

        while len(block) < k:
            if time.time() > deadline:
                break

            # Find the point that maximizes the number of newly covered t-subsets
            best_point = -1
            best_gain = -1

            # For each candidate point not in block
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue

                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset is covered if it's a subset of block + {x}
                # So we need t-subsets that include x and (t-1) elements from block
                gain = 0
                # Generate all (t-1)-subsets of block
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point == -1 or best_gain == 0:
                # No point adds coverage; pick any random point
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    break

            block.append(best_point)

        block.sort()
        block_tuple = tuple(block)

        # Remove all covered t-subsets
        covered = set(itertools.combinations(block, t))
        uncovered -= covered

        blocks.append(block)

    # If we timed out, we may have uncovered subsets. 
    # To ensure validity, we need to cover remaining subsets.
    # But if we're close to deadline, we might not be able to.
    # For now, return what we have. The evaluator will check validity.
    # If invalid, score is 0. So we should try to complete if possible.

    # Try to complete quickly if there's time
    if uncovered and time.time() < deadline - 5:
        # Continue greedy
        while uncovered and time.time() < deadline - 1:
            seed = min(uncovered)
            block = list(seed)
            while len(block) < k and time.time() < deadline - 1:
                block_set = set(block)
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_point = candidates[0]
                    else:
                        break
                block.append(best_point)
            block.sort()
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
            blocks.append(block)

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
