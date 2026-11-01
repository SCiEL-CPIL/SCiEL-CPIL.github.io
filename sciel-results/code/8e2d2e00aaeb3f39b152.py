"""Covering design C(19,8,5) - optimized greedy with bitmask acceleration and multi-restart."""
import itertools
import time
import random
import sys
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    # Precompute all t-subsets and encode as bitmasks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)

    # Map each t-subset to a bitmask of its elements
    sub_bitmasks = [0] * n_subs
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for x in sub:
            mask |= (1 << x)
        sub_bitmasks[i] = mask

    # For each point, precompute which t-subsets contain it
    point_to_subs = [[] for _ in range(v)]
    for i, mask in enumerate(sub_bitmasks):
        for x in range(v):
            if mask & (1 << x):
                point_to_subs[x].append(i)

    # Convert point_to_subs to numpy arrays for faster iteration
    point_to_subs_np = [np.array(lst, dtype=np.int32) for lst in point_to_subs]

    # Precompute sub_bitmasks as numpy array
    sub_bitmasks_np = np.array(sub_bitmasks, dtype=np.int64)

    best_blocks = None
    best_count = n_subs + 1  # worst case

    # Multi-restart randomized greedy
    max_restarts = 300
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Track uncovered t-subsets using a boolean array
        uncovered = np.ones(n_subs, dtype=np.bool_)
        uncovered_count = n_subs

        blocks = []
        block_set_cache = set()

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Find an uncovered subset - use weighted random selection favoring "hard" subsets
            found = False
            for _ in range(50):
                idx = rng.randint(0, n_subs - 1)
                if uncovered[idx]:
                    found = True
                    break
            if not found:
                # Fallback: scan
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) > 0:
                    idx = uncovered_indices[0]
                    found = True
            if not found:
                break

            # Start block with elements of this t-subset
            start_sub = all_t_subs[idx]
            block_set = set(start_sub)
            block_mask = sub_bitmasks[idx]

            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block_set) < k:
                if time.time() > deadline:
                    break

                best_gain = -1
                best_point = None
                second_best_gain = -1
                second_best_point = None

                for x in range(v):
                    if x in block_set:
                        continue

                    # Use numpy for faster gain computation
                    s_indices = point_to_subs_np[x]
                    if len(s_indices) == 0:
                        continue
                    
                    # Get uncovered subsets containing x
                    uncovered_mask = uncovered[s_indices]
                    if not np.any(uncovered_mask):
                        continue
                    
                    # Filter to only uncovered ones
                    s_indices_uncovered = s_indices[uncovered_mask]
                    
                    # Check which ones would be covered by adding x
                    # Condition: sub_bitmasks[s_idx] & ~(block_mask | (1 << x)) == 0
                    x_bit = 1 << x
                    masks = sub_bitmasks_np[s_indices_uncovered]
                    # Vectorized check
                    covered_mask = (masks & ~(block_mask | x_bit)) == 0
                    gain = np.sum(covered_mask)

                    if gain > best_gain:
                        second_best_gain = best_gain
                        second_best_point = best_point
                        best_gain = gain
                        best_point = x
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = x

                if best_point is None or best_gain <= 0:
                    # No point adds coverage; pick a random one to complete the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break

                block_set.add(best_point)
                block_mask |= (1 << best_point)

            # Convert block to sorted list
            block = sorted(block_set)
            block_tuple = tuple(block)

            # Check if this block is a duplicate
            if block_tuple in block_set_cache:
                continue

            block_set_cache.add(block_tuple)
            blocks.append(block)

            # Update uncovered: mark all t-subsets contained in this block as covered
            # Vectorized update
            covered_by_block = (sub_bitmasks_np & ~block_mask) == 0
            newly_covered = uncovered & covered_by_block
            uncovered_count -= np.sum(newly_covered)
            uncovered[newly_covered] = False

            if uncovered_count == 0:
                break

        # Check if we have a complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

            # If we're close to the known best, we can stop early
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: return a trivial (but likely huge) covering
        uncovered_set = set(range(n_subs))
        blocks = []
        for idx in sorted(uncovered_set):
            if time.time() > deadline:
                break
            block = list(all_t_subs[idx])
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
