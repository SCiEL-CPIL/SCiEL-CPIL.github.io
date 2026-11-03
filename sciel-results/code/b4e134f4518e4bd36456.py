import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # Build block_to_triple_list: list of lists of triple indices for each block
    block_to_triple_list = [list(block_triples[i]) for i in range(n_blocks)]

    # Build triple_to_block_list: list of lists of block indices for each triple
    triple_to_block_list = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tj in block_to_triple_list[i]:
            triple_to_block_list[tj].append(i)

    # Precompute block_triple_sets as frozensets for fast subset checks
    block_triple_sets = [frozenset(block_to_triple_list[i]) for i in range(n_blocks)]

    # Triple coverage frequency
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)

    # Precompute block weights
    block_weights = np.sum(triple_weights[block_triples], axis=1)

    def greedy_cover_density(uncovered_init, density_alpha=1.0, use_weighted=True, random_tie=False, tie_temp=0.0):
        """Run greedy set cover with coverage-density weighting.
        density_alpha controls the penalty for overlap with already-selected blocks.
        """
        uncovered = uncovered_init.copy()
        selected = []
        selected_triple_count = np.zeros(n_triples, dtype=np.int32)  # how many selected blocks cover each triple

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if use_weighted:
                # Base gain: sum of weights of uncovered triples in block
                base_gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                base_gains = np.sum(uncovered[block_triples], axis=1)

            # Density penalty: penalize blocks that overlap heavily with already-selected blocks
            if selected and density_alpha > 0:
                # For each block, count how many of its triples are already covered by selected blocks
                overlap = np.sum(selected_triple_count[block_triples] > 0, axis=1).astype(np.float64)
                # Density score: base_gain / (1 + alpha * overlap / c_k_t)
                density_gains = base_gains / (1.0 + density_alpha * overlap / c_k_t)
            else:
                density_gains = base_gains

            max_gain = np.max(density_gains)
            if max_gain == 0:
                break

            if random_tie:
                if tie_temp > 0:
                    exp_gains = np.exp(density_gains / tie_temp - max_gain / tie_temp)
                    total = np.sum(exp_gains)
                    if total < 1e-300:
                        break
                    probs = exp_gains / total
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    candidates = np.where(np.abs(density_gains - max_gain) < 1e-10)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(density_gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            selected_triple_count[block_triples[best_idx]] += 1

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices

        remaining = list(selected_indices)
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            for b in remaining:
                if time.time() > deadline:
                    break
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    changed = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

        return remaining

    def local_search_swap(selected_indices):
        """Max-SAT style local search with coverage-density scoring for swaps."""
        remaining = list(selected_indices)
        n = len(remaining)
        selected_set = set(remaining)

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        def try_remove(pos):
            nonlocal n
            b = remaining[pos]
            for tj in block_to_triple_list[b]:
                if cover_count[tj] <= 1:
                    return False
            for tj in block_to_triple_list[b]:
                cover_count[tj] -= 1
            remaining.pop(pos)
            selected_set.discard(b)
            n -= 1
            return True

        def try_2for1(pos1, pos2):
            """Try removing blocks at pos1, pos2 and adding one block with best density score."""
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            # Vectorized unique-triples computation
            t1 = block_triples[b1]
            t2 = block_triples[b2]
            combined = np.concatenate([t1, t2])
            cc = cover_count[combined]
            unique_mask = (cc == 1)
            # Deduplicate: a triple appearing in both blocks with count 1 is still unique
            # (it becomes count 0 after removing both, so it must be re-covered)
            unique_arr = combined[unique_mask]
            if len(unique_arr) > 0:
                unique_arr, _ = np.unique(unique_arr, return_index=True)

            if len(unique_arr) == 0:
                cover_count[t1] -= 1
                cover_count[t2] -= 1
                hi = max(pos1, pos2)
                lo = min(pos1, pos2)
                remaining.pop(hi)
                remaining.pop(lo)
                selected_set.discard(b1)
                selected_set.discard(b2)
                n -= 2
                return True

            if len(unique_arr) > c_k_t:
                return False

            ut_set = frozenset(unique_arr.tolist())

            # Find best replacement block by density score
            best_nb = -1
            best_score = -1
            tried = set()

            # Sample unselected blocks
            n_unselected = n_blocks - n
            n_samples = min(300, n_unselected)
            for _ in range(n_samples):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set or nb in tried:
                    continue
                tried.add(nb)
                if ut_set.issubset(block_triple_sets[nb]):
                    # Vectorized scoring
                    nb_triples = block_triples[nb]
                    new_triples = np.sum(cover_count[nb_triples] == 0)
                    overlap_with_selected = np.sum(cover_count[nb_triples] > 0)
                    score = new_triples - 0.1 * overlap_with_selected
                    if score > best_score:
                        best_score = score
                        best_nb = nb

            # If not found in random sampling, do targeted search for small unique_triples
            if best_nb == -1 and len(unique_arr) <= 4:
                for nb in range(n_blocks):
                    if time.time() > deadline - 2.0:
                        return False
                    if nb in selected_set or nb in tried:
                        continue
                    if ut_set.issubset(block_triple_sets[nb]):
                        nb_triples = block_triples[nb]
                        new_triples = np.sum(cover_count[nb_triples] == 0)
                        overlap_with_selected = np.sum(cover_count[nb_triples] > 0)
                        score = new_triples - 0.1 * overlap_with_selected
                        if score > best_score:
                            best_score = score
                            best_nb = nb

            if best_nb == -1:
                return False

            # Execute the swap (vectorized)
            cover_count[t1] -= 1
            cover_count[t2] -= 1
            cover_count[block_triples[best_nb]] += 1
            selected_set.discard(b1)
            selected_set.discard(b2)
            selected_set.add(best_nb)
            hi = max(pos1, pos2)
            lo = min(pos1, pos2)
            remaining.pop(hi)
            remaining.pop(lo)
            remaining.append(best_nb)
            n -= 1
            return True

        def try_1for1(pos):
            """Try removing block at pos and adding one replacement block."""
            nonlocal n
            b = remaining[pos]
            t_b = block_triples[b]
            # Triples uniquely covered by this block
            cc = cover_count[t_b]
            unique_mask = (cc == 1)
            if not np.any(unique_mask):
                # Can remove without replacement
                cover_count[t_b] -= 1
                remaining.pop(pos)
                selected_set.discard(b)
                n -= 1
                return True

            unique_arr = t_b[unique_mask]
            if len(unique_arr) > c_k_t:
                return False

            ut_set = frozenset(unique_arr.tolist())

            best_nb = -1
            best_score = -1
            tried = set()

            n_unselected = n_blocks - n
            n_samples = min(200, n_unselected)
            for _ in range(n_samples):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set or nb in tried:
                    continue
                tried.add(nb)
                if ut_set.issubset(block_triple_sets[nb]):
                    nb_triples = block_triples[nb]
                    new_triples = np.sum(cover_count[nb_triples] == 0)
                    overlap_with_selected = np.sum(cover_count[nb_triples] > 0)
                    score = new_triples - 0.1 * overlap_with_selected
                    if score > best_score:
                        best_score = score
                        best_nb = nb

            if best_nb == -1 and len(unique_arr) <= 3:
                for nb in range(n_blocks):
                    if time.time() > deadline - 2.0:
                        return False
                    if nb in selected_set or nb in tried:
                        continue
                    if ut_set.issubset(block_triple_sets[nb]):
                        nb_triples = block_triples[nb]
                        new_triples = np.sum(cover_count[nb_triples] == 0)
                        overlap_with_selected = np.sum(cover_count[nb_triples] > 0)
                        score = new_triples - 0.1 * overlap_with_selected
                        if score > best_score:
                            best_score = score
                            best_nb = nb

            if best_nb == -1:
                return False

            cover_count[t_b] -= 1
            cover_count[block_triples[best_nb]] += 1
            selected_set.discard(b)
            selected_set.add(best_nb)
            remaining[pos] = best_nb
            return True

        # Main loop
        while time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Try single removals
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                if try_remove(i):
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Try 2-for-1 swaps
            n_pairs_to_try = min(400, n * (n - 1) // 2)
            found_swap = False
            for _ in range(n_pairs_to_try):
                if time.time() > deadline - 2.0:
                    break
                i, j = random.sample(range(n), 2)
                if try_2for1(i, j):
                    found_swap = True
                    break

            if found_swap:
                continue

            # Phase 3: Try 1-for-1 swaps (replace one block with another)
            n_1for1_to_try = min(300, n)
            found_1for1 = False
            for _ in range(n_1for1_to_try):
                if time.time() > deadline - 2.0:
                    break
                i = random.randrange(n)
                if try_1for1(i):
                    found_1for1 = True
                    break

            if found_1for1:
                continue

            # Stochastic escape
            extra_tries = min(200, n * (n - 1) // 2)
            for _ in range(extra_tries):
                if time.time() > deadline - 2.0:
                    break
                i, j = random.sample(range(n), 2)
                if try_2for1(i, j):
                    improved = True
                    break

            if not improved:
                break

        return remaining

    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Strategy 1: Standard greedy (unweighted, no density)
    selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=0.0, use_weighted=False, random_tie=False)
    try_solution(selected)

    # Strategy 2: Weighted greedy (no density)
    selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=0.0, use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 3: Weighted greedy with density alpha=0.5
    selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=0.5, use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 4: Weighted greedy with density alpha=1.0
    selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=1.0, use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 5-18: Random tie-breaking restarts with varying parameters
    for restart in range(14):
        if time.time() > deadline - 45.0:
            break

        use_weighted = (restart % 2 == 0)
        if restart < 4:
            density_alpha = 0.0
        elif restart < 8:
            density_alpha = 0.5
        elif restart < 12:
            density_alpha = 1.0
        else:
            density_alpha = 2.0

        if restart < 7:
            tie_temp = 0.0
        else:
            tie_temp = 0.3 + (restart - 7) * 0.4

        selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=density_alpha,
                                        use_weighted=use_weighted, random_tie=True, tie_temp=tie_temp)
        try_solution(selected)

    # Local search on the best solution found so far
    if best_solution is not None and time.time() < deadline - 15.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        improved = local_search_swap(best_indices)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]

    # Additional greedy restarts with local search if time remains
    if time.time() < deadline - 30.0:
        for extra_restart in range(8):
            if time.time() > deadline - 20.0:
                break
            # Vary the greedy strategy
            if extra_restart % 4 == 0:
                selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=0.0, use_weighted=False, random_tie=True, tie_temp=0.0)
            elif extra_restart % 4 == 1:
                selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=0.5, use_weighted=True, random_tie=True, tie_temp=0.5)
            elif extra_restart % 4 == 2:
                selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=1.0, use_weighted=True, random_tie=True, tie_temp=1.5)
            else:
                selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=2.0, use_weighted=True, random_tie=True, tie_temp=3.0)
            try_solution(selected)

        # Final local search on best
        if best_solution is not None and time.time() < deadline - 5.0:
            block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_to_idx[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    # Final validation and return
    if best_solution is None:
        return [list(b) for b in blocks[:100]]

    return best_solution
