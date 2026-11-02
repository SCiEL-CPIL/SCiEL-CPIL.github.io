"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute: for each block (k-subset), which t-subsets it covers
    # We'll generate blocks on-the-fly but cache their t-coverage
    block_t_cache = {}

    def get_block_tsubs(block):
        """Get list of t-subset indices covered by a k-block."""
        key = tuple(sorted(block))
        if key not in block_t_cache:
            block_t_cache[key] = [t_sub_to_idx[tuple(sorted(s))] for s in itertools.combinations(key, t)]
        return block_t_cache[key]

    best_blocks = None
    best_count = float('inf')

    n_restarts = 30

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        # Phase 1: Greedy initialization with Lagrangian-guided scoring
        covered = [False] * n_t
        covered_count = 0
        blocks = []

        # Lagrangian multipliers for uncovered t-subsets
        # lambda[i] = penalty weight for t-subset i being uncovered
        lam = [1.0] * n_t

        while covered_count < n_t and time_ok():
            # Compute effective gain for each candidate block
            # using current Lagrangian weights
            # Score = sum of lambda[j] for uncovered t-subsets j covered by block

            # Find uncovered t-subsets
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            # Weighted sampling of seed t-subsets based on Lagrangian weights
            # Higher lambda = more important to cover
            if len(uncovered_indices) <= 30:
                candidates = uncovered_indices
            else:
                weights = [lam[idx] for idx in uncovered_indices]
                total_w = sum(weights)
                if total_w == 0:
                    candidates = [random.choice(uncovered_indices)]
                else:
                    n_samples = min(15, len(uncovered_indices))
                    candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                candidates.append(uncovered_indices[i])
                                break
                    if not candidates:
                        candidates = [random.choice(uncovered_indices)]

            best_block = None
            best_lagrangian_gain = -1

            for seed_idx in candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily with Lagrangian-weighted gain
                while len(block_set) < k and time_ok():
                    best_x = None
                    best_x_gain = -1

                    block_tuple = tuple(sorted(block_set))
                    candidates_pts = [x for x in all_points if x not in block_set]
                    random.shuffle(candidates_pts)

                    for x in candidates_pts:
                        # Compute Lagrangian-weighted gain of adding x
                        gain = 0
                        for rest in itertools.combinations(block_tuple, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += lam[idx]
                        tie = random.random() * 0.001
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x

                    if best_x is None:
                        break
                    block_set.add(best_x)

                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Calculate total Lagrangian-weighted gain
                lagrangian_gain = 0
                for idx in get_block_tsubs(block):
                    if not covered[idx]:
                        lagrangian_gain += lam[idx]

                if lagrangian_gain > best_lagrangian_gain:
                    best_lagrangian_gain = lagrangian_gain
                    best_block = block

            if best_block is None or best_lagrangian_gain <= 0:
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))

            # Apply the block
            newly_covered = 0
            for idx in get_block_tsubs(best_block):
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    newly_covered += 1
            blocks.append(best_block)

            # Update Lagrangian multipliers: increase for still-uncovered, decrease for covered
            for idx in range(n_t):
                if covered[idx]:
                    lam[idx] *= 0.95  # decay covered
                else:
                    lam[idx] *= 1.05  # increase uncovered

            # Normalize to prevent overflow
            max_lam = max(lam)
            if max_lam > 100:
                lam = [x / max_lam for x in lam]

        # Phase 2: WalkSAT-style stochastic local search
        # Flip block decisions to reduce total number while maintaining coverage
        if time_ok() and covered_count == n_t:
            # Count how many blocks cover each t-subset
            tsub_block_count = [0] * n_t
            block_tsubs_list = []
            for b in blocks:
                tsubs = get_block_tsubs(b)
                block_tsubs_list.append(tsubs)
                for idx in tsubs:
                    tsub_block_count[idx] += 1

            # Local search: try removing blocks that are redundant
            # Then try swapping blocks to reduce count
            n_local_iters = 500
            for it in range(n_local_iters):
                if not time_ok():
                    break

                # Try to remove a redundant block
                removable_idx = -1
                for i in range(len(blocks)):
                    removable = True
                    for idx in block_tsubs_list[i]:
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        removable_idx = i
                        break

                if removable_idx >= 0:
                    # Remove this block
                    for idx in block_tsubs_list[removable_idx]:
                        tsub_block_count[idx] -= 1
                    blocks.pop(removable_idx)
                    block_tsubs_list.pop(removable_idx)
                    continue

                # Try a swap: remove one block, add a better one
                # Pick a random block to try removing
                if len(blocks) > 10:
                    remove_idx = random.randint(0, len(blocks) - 1)
                    removed_block = blocks[remove_idx]
                    removed_tsubs = block_tsubs_list[remove_idx]

                    # Temporarily remove
                    for idx in removed_tsubs:
                        tsub_block_count[idx] -= 1

                    # Check if any t-subset becomes uncovered
                    uncovered_after_remove = []
                    for idx in removed_tsubs:
                        if tsub_block_count[idx] == 0:
                            uncovered_after_remove.append(idx)

                    if not uncovered_after_remove:
                        # Safe to remove without replacement
                        blocks.pop(remove_idx)
                        block_tsubs_list.pop(remove_idx)
                        continue
                    else:
                        # Need to find a replacement block that covers all uncovered
                        # Try to find a block that covers the most uncovered t-subsets
                        best_repl = None
                        best_repl_gain = -1

                        # Sample some candidate blocks
                        n_candidates = 20
                        for _ in range(n_candidates):
                            if not time_ok():
                                break
                            cand_set = set(random.sample(all_points, k))
                            cand = tuple(sorted(cand_set))
                            cand_tsubs = get_block_tsubs(cand)

                            # Count how many uncovered t-subsets this covers
                            gain = sum(1 for idx in cand_tsubs if idx in set(uncovered_after_remove))

                            if gain > best_repl_gain:
                                best_repl_gain = gain
                                best_repl = (cand, cand_tsubs)

                        if best_repl and best_repl_gain >= len(uncovered_after_remove):
                            # Accept the swap
                            blocks.append(best_repl[0])
                            block_tsubs_list.append(best_repl[1])
                            for idx in best_repl[1]:
                                tsub_block_count[idx] += 1
                        else:
                            # Revert
                            for idx in removed_tsubs:
                                tsub_block_count[idx] += 1
                            blocks.insert(remove_idx, removed_block)
                            block_tsubs_list.insert(remove_idx, removed_tsubs)

        # Phase 3: Final deletion pass
        if time_ok():
            tsub_block_count = [0] * n_t
            block_tsubs_list = []
            for b in blocks:
                tsubs = get_block_tsubs(b)
                block_tsubs_list.append(tsubs)
                for idx in tsubs:
                    tsub_block_count[idx] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for idx in block_tsubs_list[i]:
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs_list[i]:
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
                        block_tsubs_list.pop(i)
                        changed = True
                        break

        # Verify coverage
        if time_ok():
            final_covered = [False] * n_t
            for b in blocks:
                for idx in get_block_tsubs(b):
                    final_covered[idx] = True
            if all(final_covered):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            else:
                # Coverage broken, discard this restart
                pass
        else:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)

        # Early stop if we've done very well
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
