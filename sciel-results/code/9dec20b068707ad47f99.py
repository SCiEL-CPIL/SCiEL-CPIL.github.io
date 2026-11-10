"""
Strategy: Local Search with Simulated Annealing for Covering Design C(19,8,5)
- State: a valid cover (list of block indices).
- Moves: add a random block then prune, or remove a non-essential block.
- Fitness: number of blocks (minimize).
- Annealing: accept worse moves with decreasing probability.
- Initialization: multiple fast greedy covers with random tie-breaking.
- Key insight: local search over the space of valid covers can find
  smaller covers by exploring neighborhoods that GA crossover misses.
"""

import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    full_mask = (1 << n_t) - 1

    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}

    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= (1 << t_sub_to_idx[sub])
        block_masks[i] = mask

    # Count how many t-subsets each block covers (for quick gain estimation)
    block_gain_counts = [m.bit_count() for m in block_masks]

    def greedy_cover(seed):
        """Fast greedy set cover with random tie-breaking."""
        random.seed(seed)
        uncovered = full_mask
        selected = []

        while uncovered:
            best_idx = -1
            best_gain = -1
            # Sample a subset of blocks for speed, or all if small enough
            if n_blocks <= 20000:
                candidates = range(n_blocks)
            else:
                candidates = random.sample(range(n_blocks), min(5000, n_blocks))

            for i in candidates:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noisy = gain + random.random() * 0.3
                    if noisy > best_gain:
                        best_gain = noisy
                        best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return None

        return selected

    def prune_cover(block_indices):
        """Remove redundant blocks. Returns pruned list of indices."""
        if not block_indices:
            return list(block_indices)

        current = list(block_indices)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Count coverage of each t-subset
            counts = [0] * n_t
            for idx in current:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    ti = lsb.bit_length() - 1
                    counts[ti] += 1
                    m ^= lsb

            new_current = []
            for idx in current:
                mask = block_masks[idx]
                essential = False
                m = mask
                while m:
                    lsb = m & -m
                    ti = lsb.bit_length() - 1
                    if counts[ti] == 1:
                        essential = True
                        break
                    m ^= lsb
                if essential:
                    new_current.append(idx)
                else:
                    changed = True
            current = new_current

        return current

    def is_valid(block_indices):
        """Check if the set of blocks covers all t-subsets."""
        covered = 0
        for idx in block_indices:
            covered |= block_masks[idx]
        return (covered & full_mask) == full_mask

    def local_search(initial_cover, budget):
        """Simulated annealing local search over valid covers."""
        if not initial_cover:
            return initial_cover

        current = list(initial_cover)
        current = prune_cover(current)
        best = list(current)
        best_len = len(current)

        if not is_valid(current):
            return best

        T = 2.0  # Initial temperature
        cooling = 0.995
        min_T = 0.01

        random.shuffle(current)  # Diversify

        while time.time() < budget and T > min_T:
            move_type = random.random()

            if move_type < 0.6:
                # Add a random block, then prune
                new_block = random.randint(0, n_blocks - 1)
                if new_block not in current:
                    candidate = current + [new_block]
                    candidate = prune_cover(candidate)
                    if is_valid(candidate):
                        delta = len(candidate) - len(current)
                        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                            current = candidate
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
            elif move_type < 0.85:
                # Remove a random non-essential block
                # Find non-essential blocks
                non_essential = []
                counts = [0] * n_t
                for idx in current:
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        ti = lsb.bit_length() - 1
                        counts[ti] += 1
                        m ^= lsb

                for idx in current:
                    mask = block_masks[idx]
                    essential = False
                    m = mask
                    while m:
                        lsb = m & -m
                        ti = lsb.bit_length() - 1
                        if counts[ti] == 1:
                            essential = True
                            break
                        m ^= lsb
                    if not essential:
                        non_essential.append(idx)

                if non_essential:
                    remove_idx = random.choice(non_essential)
                    candidate = [x for x in current if x != remove_idx]
                    if is_valid(candidate):
                        delta = len(candidate) - len(current)
                        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                            current = candidate
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
            else:
                # Swap: remove one block, add another, prune
                if len(current) > 1:
                    remove_idx = random.choice(current)
                    add_idx = random.randint(0, n_blocks - 1)
                    candidate = [x for x in current if x != remove_idx]
                    if add_idx not in candidate:
                        candidate.append(add_idx)
                    candidate = prune_cover(candidate)
                    if is_valid(candidate):
                        delta = len(candidate) - len(current)
                        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                            current = candidate
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)

            T *= cooling

        return best

    # Main: generate diverse initial solutions, then local search on each
    best_overall = None
    best_len = float('inf')

    num_starts = 8
    time_per_start = (deadline - start_time) / num_starts

    for s in range(num_starts):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break

        # Greedy initialization
        seed = 12345 + s * 7919
        init = greedy_cover(seed)
        if init is None:
            break

        init = prune_cover(init)
        if not init or not is_valid(init):
            continue

        if len(init) < best_len:
            best_len = len(init)
            best_overall = list(init)

        # Local search with time budget
        ls_budget = min(time_per_start, deadline - time.time() - 0.5)
        if ls_budget > 0.5:
            result = local_search(init, start_time + ls_budget)
            if result and is_valid(result) and len(result) < best_len:
                best_len = len(result)
                best_overall = list(result)

    if best_overall is None:
        # Fallback: single greedy
        init = greedy_cover(42)
        if init:
            init = prune_cover(init)
            return [all_blocks[i] for i in init]
        return []

    return [all_blocks[i] for i in best_overall]
