import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    # block_triples[i] is a sorted array of triple indices covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # Triple coverage frequency
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)

    # Precompute block weights
    block_weights = np.sum(triple_weights[block_triples], axis=1)

    # Precompute block_triples as sets for fast subset checking
    block_triple_sets = [set(block_triples[i].tolist()) for i in range(n_blocks)]

    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, tie_temp=0.0):
        """Run greedy set cover, optionally weighted with optional random tie-breaking."""
        uncovered = uncovered_init.copy()
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tie:
                if tie_temp > 0:
                    exp_gains = np.exp(gains / tie_temp - max_gain / tie_temp)
                    total = np.sum(exp_gains)
                    if total < 1e-300:
                        break
                    probs = exp_gains / total
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering."""
        if not selected_indices:
            return selected_indices

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in selected_indices:
            cover_count[block_triples[b]] += 1

        remaining = list(selected_indices)
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
        """Local search: removals, 1-for-1 replacements, and 2-for-1 swaps."""
        remaining = list(selected_indices)
        n = len(remaining)
        selected_set = set(remaining)

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        def try_remove(pos):
            nonlocal n
            b = remaining[pos]
            for tj in block_triples[b]:
                if cover_count[tj] <= 1:
                    return False
            for tj in block_triples[b]:
                cover_count[tj] -= 1
            remaining.pop(pos)
            selected_set.discard(b)
            n -= 1
            return True

        def try_1for1(pos):
            """Try replacing block at pos with a different block that covers all its critical triples."""
            nonlocal n
            b = remaining[pos]
            # Find triples that would become uncovered if we remove b
            critical = []
            for tj in block_triples[b]:
                if cover_count[tj] == 1:
                    critical.append(tj)
            if not critical:
                # Can just remove
                return try_remove(pos)

            crit_set = set(critical)
            # Sample unselected blocks to find one covering all critical triples
            n_unselected = n_blocks - n
            if n_unselected <= 0:
                return False

            for _ in range(min(300, n_unselected)):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set:
                    continue
                if crit_set.issubset(block_triple_sets[nb]):
                    # Do the replacement
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    for tj in block_triples[nb]:
                        cover_count[tj] += 1
                    selected_set.discard(b)
                    selected_set.add(nb)
                    remaining[pos] = nb
                    return True
            return False

        def try_2for1(pos1, pos2):
            """Try removing blocks at pos1, pos2 and adding one block."""
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            unique_triples = []
            seen = set()
            for tj in block_triples[b1]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)
            for tj in block_triples[b2]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)

            if not unique_triples:
                for tj in block_triples[b1]:
                    cover_count[tj] -= 1
                for tj in block_triples[b2]:
                    cover_count[tj] -= 1
                hi = max(pos1, pos2)
                lo = min(pos1, pos2)
                remaining.pop(hi)
                remaining.pop(lo)
                selected_set.discard(b1)
                selected_set.discard(b2)
                n -= 2
                return True

            if len(unique_triples) > c_k_t:
                return False

            ut_set = set(unique_triples)
            n_unselected = n_blocks - n
            if n_unselected <= 0:
                return False

            # Sample unselected blocks
            for _ in range(min(400, n_unselected)):
                if time.time() > deadline - 2.0:
                    return False
                nb = random.randrange(n_blocks)
                if nb in selected_set:
                    continue
                if ut_set.issubset(block_triple_sets[nb]):
                    for tj in block_triples[b1]:
                        cover_count[tj] -= 1
                    for tj in block_triples[b2]:
                        cover_count[tj] -= 1
                    for tj in block_triples[nb]:
                        cover_count[tj] += 1
                    selected_set.discard(b1)
                    selected_set.discard(b2)
                    selected_set.add(nb)
                    hi = max(pos1, pos2)
                    lo = min(pos1, pos2)
                    remaining.pop(hi)
                    remaining.pop(lo)
                    remaining.append(nb)
                    n -= 1
                    return True
            return False

        # Main loop
        while time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Single removals
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                if try_remove(i):
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: 1-for-1 replacements (new)
            for _ in range(min(50, n)):
                if time.time() > deadline - 2.0:
                    break
                pos = random.randrange(n)
                if try_1for1(pos):
                    improved = True
                    break

            if improved:
                continue

            # Phase 3: 2-for-1 swaps
            n_pairs_to_try = min(300, n * (n - 1) // 2)
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

            # Stochastic escape: more random pairs
            extra_tries = min(150, n * (n - 1) // 2)
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

    # Run multiple greedy passes
    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    try_solution(selected)

    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 3-14: Random tie-breaking restarts
    for restart in range(12):
        if time.time() > deadline - 30.0:
            break
        use_weighted = (restart % 2 == 0)
        tie_temp = 0.0 if restart < 6 else (0.5 + restart * 0.3)
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted,
                                random_tie=True, tie_temp=tie_temp)
        try_solution(selected)

    # Local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        improved = local_search_swap(best_indices)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]

    # Additional greedy restarts with local search if time remains
    if time.time() < deadline - 20.0:
        for extra_restart in range(8):
            if time.time() > deadline - 10.0:
                break
            if extra_restart % 3 == 0:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=True, tie_temp=0.0)
            elif extra_restart % 3 == 1:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=True, tie_temp=1.0)
            else:
                selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=True, tie_temp=2.0)
            try_solution(selected)

        # Final local search on best
        if best_solution is not None and time.time() < deadline - 3.0:
            block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_to_idx[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        return [list(b) for b in blocks[:100]]

    return best_solution
