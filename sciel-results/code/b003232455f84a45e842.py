import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    num_5 = len(all_5_subsets)  # 792

    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)  # 924

    # Precompute bitmasks for each 6-subset (which 5-subsets it covers)
    block_masks = [0] * num_6
    for j, b in enumerate(all_6_subsets):
        mask = 0
        for s in combinations(b, 5):
            idx = index_5[s]
            mask |= (1 << idx)
        block_masks[j] = mask

    # For each 5-subset, list of 6-subsets that contain it
    subsets_5_to_6 = [[] for _ in range(num_5)]
    for j, b in enumerate(all_6_subsets):
        b_set = set(b)
        for i, s5 in enumerate(all_5_subsets):
            if all(x in b_set for x in s5):
                subsets_5_to_6[i].append(j)

    # Best result found so far
    best_result = []

    # Try multiple random restarts with different initial blocks and randomization seeds
    import random

    # Generate candidate initial blocks (those containing point 0 to break symmetry)
    initial_block_candidates = [b for b in all_6_subsets if 0 in b]
    # That's C(11,5) = 462 blocks

    for restart in range(20):
        if time.time() > deadline - 5:
            break

        rng = random.Random(restart * 12345 + 67890)

        # Pick a random initial block containing 0
        first_block = rng.choice(initial_block_candidates)
        first_block_idx = all_6_subsets.index(first_block)

        blocks = [first_block]
        covered_mask = block_masks[first_block_idx]
        uncovered = [i for i in range(num_5) if not (covered_mask >> i) & 1]

        def backtrack(covered_mask, uncovered):
            nonlocal blocks, best_result

            if len(blocks) == 132:
                return True

            if time.time() > deadline - 5:
                return False

            if not uncovered:
                return False

            # MRV: find uncovered 5-subset with fewest valid candidate blocks
            min_candidates = float('inf')
            target_idx = -1
            valid_cands = []

            for i in uncovered:
                cands = subsets_5_to_6[i]
                valid = [j for j in cands if (covered_mask & block_masks[j]) == 0]
                if len(valid) < min_candidates:
                    min_candidates = len(valid)
                    target_idx = i
                    valid_cands = valid
                    if min_candidates == 0:
                        break

            if target_idx == -1 or min_candidates == 0:
                # Update best_result if current solution is longer
                if len(blocks) > len(best_result):
                    best_result = [list(b) for b in blocks]
                return False

            # Randomize candidate order
            rng.shuffle(valid_cands)

            for j in valid_cands:
                blocks.append(all_6_subsets[j])
                new_covered_mask = covered_mask | block_masks[j]
                new_uncovered = [k for k in uncovered if not (new_covered_mask >> k) & 1]

                if backtrack(new_covered_mask, new_uncovered):
                    if len(blocks) > len(best_result):
                        best_result = [list(b) for b in blocks]
                    return True

                blocks.pop()

            # Update best_result if current solution is longer
            if len(blocks) > len(best_result):
                best_result = [list(b) for b in blocks]
            return False

        if backtrack(covered_mask, uncovered):
            break  # Found full solution

    return best_result
