import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin from 120s limit

    # Precompute all targets (t-subsets) and all blocks (k-subsets)
    all_targets = list(itertools.combinations(range(v), t))
    target_index = {tup: i for i, tup in enumerate(all_targets)}
    num_targets = len(all_targets)

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute coverage: for each block, which targets it covers
    # Use a set of target indices for each block
    block_coverage = []
    for b in all_blocks:
        covered = set()
        for sub in itertools.combinations(b, t):
            covered.add(target_index[sub])
        block_coverage.append(covered)

    # Precompute for each target, which blocks cover it (inverse index)
    target_to_blocks = [set() for _ in range(num_targets)]
    for i, cov in enumerate(block_coverage):
        for t_idx in cov:
            target_to_blocks[t_idx].add(i)

    def greedy_construct(seed_offset=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = random.Random(seed_offset)
        uncovered = set(range(num_targets))
        blocks_idx = []
        used = set()

        while uncovered:
            if time.time() > deadline - 15:
                break

            best_idx = -1
            best_gain = -1
            best_tie_breaker = None

            # Randomize iteration order
            indices = list(range(num_blocks))
            rng.shuffle(indices)

            for i in indices:
                if i in used:
                    continue
                gain = len(block_coverage[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_tie_breaker = rng.random()
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: use random value
                    if best_tie_breaker is None or rng.random() > best_tie_breaker:
                        best_idx = i
                        best_tie_breaker = rng.random()

            if best_idx == -1 or best_gain == 0:
                break

            blocks_idx.append(best_idx)
            used.add(best_idx)
            uncovered -= block_coverage[best_idx]

        return blocks_idx

    def prune_redundant(blocks_idx):
        """Remove redundant blocks (those whose coverage is fully covered by others)."""
        if not blocks_idx:
            return blocks_idx

        cover_count = np.zeros(num_targets, dtype=np.int32)
        for idx in blocks_idx:
            for t in block_coverage[idx]:
                cover_count[t] += 1

        changed = True
        while changed:
            if time.time() > deadline - 10:
                break
            changed = False
            for i in range(len(blocks_idx) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                idx = blocks_idx[i]
                redundant = True
                for t in block_coverage[idx]:
                    if cover_count[t] == 1:
                        redundant = False
                        break
                if redundant:
                    for t in block_coverage[idx]:
                        cover_count[t] -= 1
                    blocks_idx.pop(i)
                    changed = True

        return blocks_idx

    def local_search(blocks_idx, seed_offset=0):
        """Randomized local search with smooth penalty objective."""
        rng = random.Random(seed_offset + 10000)
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(num_targets, dtype=np.int32)
        for idx in blocks_idx:
            for t in block_coverage[idx]:
                cover_count[t] += 1

        # Smooth objective: sum of max(0, 1 - cover_count[t]) + lambda * len(blocks)
        # We want to minimize this. Uncovered targets contribute 1 each.
        # lambda is a small penalty to encourage fewer blocks when coverage is maintained.
        lambda_penalty = 0.01

        def objective(counts, num_blocks_used):
            # Sum of uncovered targets
            uncov = np.sum(np.maximum(0, 1 - counts))
            return uncov + lambda_penalty * num_blocks_used

        current_obj = objective(cover_count, len(blocks_idx))
        best_blocks = list(blocks_idx)
        best_obj = current_obj

        # Precompute for each block, the set of targets it covers (as list for speed)
        block_cov_lists = [list(cov) for cov in block_coverage]

        max_iterations = 50000
        no_improve_count = 0
        stall_threshold = 5000

        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break

            if no_improve_count > stall_threshold:
                break

            # Pick a random block to try removing
            i = rng.randint(0, len(blocks_idx) - 1)
            idx = blocks_idx[i]

            # Try removing this block
            new_counts = cover_count.copy()
            for t in block_cov_lists[idx]:
                new_counts[t] -= 1

            new_obj = objective(new_counts, len(blocks_idx) - 1)

            if new_obj < current_obj:
                # Accept: block is redundant or reduces objective
                cover_count = new_counts
                blocks_idx.pop(i)
                current_obj = new_obj
                no_improve_count = 0

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_blocks = list(blocks_idx)
            else:
                no_improve_count += 1

            # Try adding a random block
            if len(blocks_idx) < num_blocks:
                new_idx = rng.randint(0, num_blocks - 1)
                if new_idx not in set(blocks_idx):
                    new_counts = cover_count.copy()
                    for t in block_cov_lists[new_idx]:
                        new_counts[t] += 1
                    new_obj = objective(new_counts, len(blocks_idx) + 1)

                    # Accept if it improves or is neutral (to explore)
                    if new_obj <= current_obj + 0.5:
                        cover_count = new_counts
                        blocks_idx.append(new_idx)
                        current_obj = new_obj
                        no_improve_count = 0

                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_blocks = list(blocks_idx)
                    else:
                        no_improve_count += 1

            # Try swapping: remove one block, add another
            if len(blocks_idx) >= 1:
                i_remove = rng.randint(0, len(blocks_idx) - 1)
                idx_remove = blocks_idx[i_remove]
                idx_add = rng.randint(0, num_blocks - 1)

                if idx_add != idx_remove:
                    # Check if idx_add is already in blocks
                    if idx_add not in set(blocks_idx):
                        new_counts = cover_count.copy()
                        for t in block_cov_lists[idx_remove]:
                            new_counts[t] -= 1
                        for t in block_cov_lists[idx_add]:
                            new_counts[t] += 1

                        new_obj = objective(new_counts, len(blocks_idx))

                        if new_obj < current_obj:
                            # Accept swap
                            cover_count = new_counts
                            blocks_idx[i_remove] = idx_add
                            current_obj = new_obj
                            no_improve_count = 0

                            if current_obj < best_obj:
                                best_obj = current_obj
                                best_blocks = list(blocks_idx)
                        else:
                            no_improve_count += 1

        return best_blocks

    # Main search: multiple greedy restarts + local search
    best_solution = None
    best_len = float('inf')

    # Run several greedy restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        blocks_idx = greedy_construct(r)
        blocks_idx = prune_redundant(blocks_idx)
        blocks_idx = local_search(blocks_idx, r)
        blocks_idx = prune_redundant(blocks_idx)

        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_solution = blocks_idx

    # If we have time, do more local search on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        best_solution = local_search(best_solution, 99999)
        best_solution = prune_redundant(best_solution)

    if best_solution is None:
        # Fallback: single greedy
        best_solution = greedy_construct(0)
        best_solution = prune_redundant(best_solution)

    return [list(all_blocks[i]) for i in best_solution]
