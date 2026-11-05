import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute for each block, the list of t-subset indices it covers
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_tsubs = []
    for block in all_blocks:
        tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)

    best_blocks = None
    best_count = float('inf')

    def _remove_redundant(blocks, time_limit=None):
        """Remove redundant blocks iteratively."""
        if time_limit is None:
            time_limit = deadline - 1.0
        changed = True
        while changed:
            changed = False
            if time.time() > time_limit:
                break

            # Build coverage count
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[t_sub_to_idx[sub]] += 1

            # Check each block for redundancy
            redundant_indices = []
            for i, b in enumerate(blocks):
                is_essential = False
                for sub in itertools.combinations(b, t):
                    if coverage_count[t_sub_to_idx[sub]] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_indices.append(i)

            if redundant_indices:
                idx = random.choice(redundant_indices)
                blocks.pop(idx)
                changed = True
        return blocks

    def _greedy_construct(uncovered, seed_subset=None):
        """Construct a block greedily to maximize coverage of uncovered t-subsets."""
        if seed_subset is not None:
            block = list(seed_subset)
        else:
            # Choose a random uncovered t-subset as seed
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                return None
            seed_idx = random.choice(uncovered_indices)
            block = list(all_t_subsets[seed_idx])

        block_set = set(block)
        while len(block) < k:
            best_gain = -1
            tied_candidates = []

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for rest in t_minus_1_subsets:
                    T = tuple(sorted(rest + (x,)))
                    if T in t_sub_to_idx:
                        idx = t_sub_to_idx[T]
                        if uncovered[idx]:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    tied_candidates = [x]
                elif gain == best_gain and gain > -1:
                    tied_candidates.append(x)

            if not tied_candidates:
                break

            best_x = random.choice(tied_candidates)
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        return block

    def _local_search(blocks, time_limit=None):
        """Local search: try to replace blocks to reduce count."""
        if time_limit is None:
            time_limit = deadline - 5.0

        current_blocks = [list(b) for b in blocks]
        improved = True

        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(current_blocks)):
                if time.time() > time_limit:
                    break

                # Temporarily remove block i
                removed_block = current_blocks[i]
                current_blocks_without = current_blocks[:i] + current_blocks[i+1:]

                # Check if coverage is still complete
                coverage_count = np.zeros(num_t, dtype=np.int32)
                for b in current_blocks_without:
                    for sub in itertools.combinations(b, t):
                        coverage_count[t_sub_to_idx[sub]] += 1

                uncovered_mask = coverage_count == 0
                if not uncovered_mask.any():
                    # Block i is redundant, remove it
                    current_blocks = current_blocks_without
                    improved = True
                    break
                else:
                    # Try to cover the uncovered subsets with a new block
                    new_block = _greedy_construct(uncovered_mask)
                    if new_block is None:
                        continue

                    # Check if new block covers all uncovered
                    new_block_set = set(new_block)
                    covers_all = True
                    for sub in itertools.combinations(new_block, t):
                        pass  # We'll verify coverage after

                    new_coverage = np.zeros(num_t, dtype=np.int32)
                    for b in current_blocks_without:
                        for sub in itertools.combinations(b, t):
                            new_coverage[t_sub_to_idx[sub]] += 1
                    for sub in itertools.combinations(new_block, t):
                        new_coverage[t_sub_to_idx[sub]] += 1

                    if not (new_coverage == 0).any():
                        current_blocks = current_blocks_without + [new_block]
                        improved = True
                        break

        return current_blocks

    # Phase 1: Multiple greedy runs with randomization
    num_runs = 200
    for run_id in range(num_runs):
        if time.time() > deadline - 15.0:
            break

        random.seed(run_id * 7919 + 12345)

        uncovered = np.ones(num_t, dtype=bool)
        blocks = []

        while uncovered.any():
            block = _greedy_construct(uncovered)
            if block is None:
                break
            blocks.append(block)

            for sub in itertools.combinations(block, t):
                uncovered[t_sub_to_idx[sub]] = False

        # Remove redundant blocks
        blocks = _remove_redundant(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    # Phase 2: Local search improvement on best solution
    if best_blocks is not None and time.time() < deadline - 20.0:
        current_blocks = _local_search(best_blocks)
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]

    # Phase 3: ILS (Iterated Local Search) with perturbation
    if best_blocks is not None and time.time() < deadline - 10.0:
        current_blocks = [list(b) for b in best_blocks]
        current_count = len(current_blocks)

        for iter_id in range(100):
            if time.time() > deadline - 5.0:
                break

            # Perturbation: randomly replace 1-2 blocks
            num_perturb = random.randint(1, 2)
            perturbed_blocks = [list(b) for b in current_blocks]

            for _ in range(num_perturb):
                if len(perturbed_blocks) == 0:
                    break
                idx = random.randrange(len(perturbed_blocks))
                block_tsubs_list = list(itertools.combinations(perturbed_blocks[idx], t))
                if block_tsubs_list:
                    random_sub = random.choice(block_tsubs_list)
                    uncovered_for_greedy = np.ones(num_t, dtype=bool)
                    new_block = _greedy_construct(uncovered_for_greedy, random_sub)
                    if new_block is not None:
                        perturbed_blocks[idx] = new_block

            # Local search on perturbed
            improved_blocks = _local_search(perturbed_blocks, deadline - 5.0)

            if len(improved_blocks) < current_count:
                current_blocks = improved_blocks
                current_count = len(current_blocks)
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = [list(b) for b in current_blocks]

    return best_blocks
