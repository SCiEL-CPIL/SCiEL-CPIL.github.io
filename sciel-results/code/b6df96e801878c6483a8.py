import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # Compute how many blocks cover each triple (for weighted greedy)
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)

    # Precompute triple_to_blocks for efficient local search
    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1

    # Build a set of frozensets for each block's triples for fast membership
    block_triple_sets = [set(row.tolist()) for row in block_triples]

    def greedy_cover(random_tiebreak=False, weight_alpha=1.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            # Weighted gain: sum of weights of uncovered triples in block
            mask = uncovered[block_triples]  # (n_blocks, c_k_t) bool
            gains = np.sum(mask * triple_weights[block_triples], axis=1)

            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break

        return result

    def local_search(block_indices, max_iterations=30):
        """Try to replace or remove blocks to reduce count while maintaining coverage."""
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                bt_bi = block_triples[bi]

                # Check for full redundancy
                if np.all(coverage[bt_bi] > 1):
                    coverage[bt_bi] -= 1
                    current.pop(i)
                    improved = True
                    break

                # Find critical triples (coverage == 1)
                critical_mask = coverage[bt_bi] == 1
                if not np.any(critical_mask):
                    continue

                critical_triples = bt_bi[critical_mask]

                # Find candidate blocks that cover ALL critical_triples
                # Use the critical triple with fewest covering blocks
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]

                other_critical = critical_triples[critical_triples != min_ct]
                other_critical_set = set(other_critical.tolist())

                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        # Check if cand covers all other_critical using set intersection
                        if other_critical_set.issubset(block_triple_sets[cand]):
                            valid_candidates.append(cand)

                if valid_candidates:
                    # Pick candidate that maximizes additional coverage
                    best_cand = valid_candidates[0]
                    best_gain = -1
                    for vc in valid_candidates:
                        # Count how many triples in vc are currently covered by bi
                        # and would remain covered
                        gain = np.sum(coverage[block_triples[vc]] > 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = vc

                    coverage[bt_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break

            if not improved:
                break

        return current

    def simulated_annealing(block_indices, max_iters=200, start_temp=1.0, end_temp=0.01):
        """
        Simulated annealing to minimize block count.
        Moves:
          1. Remove a redundant block (always accept if valid)
          2. Swap a block with another that maintains coverage
          3. Add a block then remove a different one (net zero, may enable future removals)
        """
        if len(block_indices) == 0:
            return block_indices

        current = list(block_indices)
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in current:
            coverage[block_triples[bi]] += 1

        best = list(current)
        best_size = len(current)
        best_coverage = coverage.copy()

        rng = np.random.RandomState(42)

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            temp = start_temp * (end_temp / start_temp) ** (it / max(1, max_iters - 1))

            move_type = rng.randint(0, 3)
            new_current = list(current)
            new_coverage = coverage.copy()
            valid = True

            if move_type == 0:
                # Try to remove a random block
                if len(new_current) <= 1:
                    continue
                idx = rng.randint(0, len(new_current))
                bi = new_current[idx]
                bt_bi = block_triples[bi]
                if np.any(coverage[bt_bi] <= 1):
                    continue
                # Valid removal
                new_coverage[bt_bi] -= 1
                new_current.pop(idx)

            elif move_type == 1:
                # Swap a block with another
                if len(new_current) <= 1:
                    continue
                idx = rng.randint(0, len(new_current))
                bi = new_current[idx]
                bt_bi = block_triples[bi]

                # Find candidates that can replace bi
                critical_mask = coverage[bt_bi] == 1
                if not np.any(critical_mask):
                    # Can just remove, handled by move 0
                    continue

                critical_triples = bt_bi[critical_mask]
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]

                other_critical = critical_triples[critical_triples != min_ct]
                other_critical_set = set(other_critical.tolist())

                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        if other_critical_set.issubset(block_triple_sets[cand]):
                            valid_candidates.append(cand)

                if not valid_candidates:
                    continue

                cand = valid_candidates[rng.randint(0, len(valid_candidates))]
                new_coverage[bt_bi] -= 1
                new_coverage[block_triples[cand]] += 1
                new_current[idx] = cand

            else:
                # Add a random block, then try to remove a different one
                if len(new_current) >= n_blocks - 1:
                    continue
                # Pick a random block not already in current
                current_set = set(new_current)
                all_indices = list(range(n_blocks))
                available = [x for x in all_indices if x not in current_set]
                if not available:
                    continue
                add_idx = available[rng.randint(0, len(available))]
                new_current.append(add_idx)
                new_coverage[block_triples[add_idx]] += 1

                # Now try to remove a random block
                if len(new_current) > 1:
                    rem_idx = rng.randint(0, len(new_current))
                    rem_bi = new_current[rem_idx]
                    bt_rem = block_triples[rem_bi]
                    if np.all(new_coverage[bt_rem] > 1):
                        new_coverage[bt_rem] -= 1
                        new_current.pop(rem_idx)
                    else:
                        # Can't remove, revert the add
                        new_current.pop()
                        new_coverage[block_triples[add_idx]] -= 1
                        continue

            # Evaluate the move
            new_size = len(new_current)
            cur_size = len(current)

            if new_size < cur_size:
                # Better, always accept
                current = new_current
                coverage = new_coverage
                if new_size < best_size:
                    best = list(current)
                    best_size = new_size
                    best_coverage = coverage.copy()
            elif new_size == cur_size:
                # Same size, accept with probability based on temperature
                # Use a simple heuristic: accept if it might help future removals
                if rng.random() < 0.3:
                    current = new_current
                    coverage = new_coverage
            else:
                # Worse, accept with probability exp(-delta/temp)
                delta = new_size - cur_size
                if temp > 0:
                    prob = np.exp(-delta / temp)
                    if rng.random() < prob:
                        current = new_current
                        coverage = new_coverage

        return best

    # First greedy pass
    selected = greedy_cover(random_tiebreak=False)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_iterations=25)

    best_solution = [list(blocks[i]) for i in selected]
    best_size = len(best_solution)

    # Apply simulated annealing to the best solution
    sa_result = simulated_annealing(selected, max_iters=150, start_temp=1.0, end_temp=0.01)
    if len(sa_result) < best_size:
        best_size = len(sa_result)
        best_solution = [list(blocks[i]) for i in sa_result]

    # Multiple random restarts with varied parameters
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break

        # Vary the random seed and tie-breaking aggressiveness
        np.random.seed(r * 12345 + 7)
        random.seed(r * 54321 + 3)

        selected = greedy_cover(random_tiebreak=True)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=15)

        # Apply simulated annealing
        sa_result = simulated_annealing(selected, max_iters=100, start_temp=0.8, end_temp=0.02)
        if len(sa_result) < best_size:
            best_size = len(sa_result)
            best_solution = [list(blocks[i]) for i in sa_result]

    # Final polish: run local search on the best solution one more time
    final_indices = [blocks.index(tuple(b)) for b in best_solution]
    final_indices = local_search(final_indices, max_iterations=30)
    if len(final_indices) < best_size:
        best_size = len(final_indices)
        best_solution = [list(blocks[i]) for i in final_indices]

    return best_solution
