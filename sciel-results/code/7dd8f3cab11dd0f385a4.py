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

    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    block_to_triple_list = [list(block_triples[i]) for i in range(n_blocks)]

    triple_to_block_list = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tj in block_to_triple_list[i]:
            triple_to_block_list[tj].append(i)

    block_triple_sets = [frozenset(block_to_triple_list[i]) for i in range(n_blocks)]

    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    block_weights = np.sum(triple_weights[block_triples], axis=1)

    def greedy_cover_density(uncovered_init, density_alpha=1.0, use_weighted=True, random_tie=False, tie_temp=0.0):
        uncovered = uncovered_init.copy()
        selected = []
        selected_triple_count = np.zeros(n_triples, dtype=np.int32)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if use_weighted:
                base_gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                base_gains = np.sum(uncovered[block_triples], axis=1)

            if selected and density_alpha > 0:
                overlap = np.sum(selected_triple_count[block_triples] > 0, axis=1).astype(np.float64)
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
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            t1 = block_triples[b1]
            t2 = block_triples[b2]
            combined = np.concatenate([t1, t2])
            cc = cover_count[combined]
            unique_mask = (cc == 1)
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

            best_nb = -1
            best_score = -1
            tried = set()

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
                    nb_triples = block_triples[nb]
                    new_triples = np.sum(cover_count[nb_triples] == 0)
                    overlap_with_selected = np.sum(cover_count[nb_triples] > 0)
                    score = new_triples - 0.1 * overlap_with_selected
                    if score > best_score:
                        best_score = score
                        best_nb = nb

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
            nonlocal n
            b = remaining[pos]
            t_b = block_triples[b]
            cc = cover_count[t_b]
            unique_mask = (cc == 1)
            if not np.any(unique_mask):
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

        def try_rebuild(pos):
            """Targeted 1-for-1 rewrite: build a replacement block that MUST cover the
            unique (count==1) triples of the removed block, then greedily fill the
            remaining slots to maximize coverage of low-count triples."""
            nonlocal n
            b = remaining[pos]
            t_b = block_triples[b]
            cc = cover_count[t_b]
            unique_mask = (cc == 1)
            if not np.any(unique_mask):
                # fully redundant: just drop it
                cover_count[t_b] -= 1
                remaining.pop(pos)
                selected_set.discard(b)
                n -= 1
                return True

            unique_arr = t_b[unique_mask]
            if len(unique_arr) > c_k_t:
                return False

            # Points that appear in any unique triple are mandatory in the new block.
            mandatory = set()
            for ti in unique_arr:
                mandatory.update(triples[ti])
            if len(mandatory) > k:
                return False  # impossible to cover all critical triples in one block

            new_set = set(mandatory)
            candidates = [x for x in range(v) if x not in new_set]
            random.shuffle(candidates)

            def gain_of(x):
                g = 0
                for rest in itertools.combinations(new_set, t - 1):
                    idx = triple_to_idx.get((min(rest + (x,)),) if False else tuple(sorted(rest + (x,))))
                    if idx is not None:
                        c = cover_count[idx]
                        if c == 0:
                            g += 100
                        elif c == 1:
                            g += 1
                return g

            for x in candidates:
                if len(new_set) >= k:
                    break
                if gain_of(x) > 0:
                    new_set.add(x)
                elif len(new_set) < k:
                    new_set.add(x)

            if len(new_set) != k:
                return False

            new_block = tuple(sorted(new_set))
            if new_block == blocks[b]:
                return False

            nb = block_index[new_block]
            if nb in selected_set:
                return False

            nb_triples = block_triples[nb]
            # validity: no triple loses its last cover
            for ti in t_b:
                if cover_count[ti] == 1 and ti not in nb_triples:
                    return False

            cover_count[t_b] -= 1
            cover_count[nb_triples] += 1
            selected_set.discard(b)
            selected_set.add(nb)
            remaining[pos] = nb
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

            # Phase 3: Try random 1-for-1 swaps
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

            # Phase 4: Targeted 1-for-1 rewrites (inspiration's strongest refinement)
            n_rebuild_to_try = min(400, n)
            found_rebuild = False
            for _ in range(n_rebuild_to_try):
                if time.time() > deadline - 2.0:
                    break
                i = random.randrange(n)
                if try_rebuild(i):
                    found_rebuild = True
                    break
            if found_rebuild:
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

    # Random tie-breaking restarts with varying parameters
    for restart in range(16):
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

        if restart < 8:
            tie_temp = 0.0
        else:
            tie_temp = 0.3 + (restart - 8) * 0.4

        selected = greedy_cover_density(np.ones(n_triples, dtype=bool), density_alpha=density_alpha,
                                        use_weighted=use_weighted, random_tie=True, tie_temp=tie_temp)
        try_solution(selected)

    # Local search on the best solution found so far
    if best_solution is not None and time.time() < deadline - 15.0:
        block_index = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_index[tuple(b)] for b in best_solution]
        improved = local_search_swap(best_indices)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]

    # Additional greedy restarts with local search if time remains
    if time.time() < deadline - 30.0:
        for extra_restart in range(10):
            if time.time() > deadline - 20.0:
                break
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
            block_index = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_index[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        return [list(b) for b in blocks[:100]]

    return best_solution
