import numpy as np
import itertools
import time
import random
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    c_k_t = comb(k, t)

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
        block_triples[i] = np.sort(block_triples[i])

    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1

    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)

    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1

    block_sets = [set(block) for block in all_blocks]
    block_tuples = [tuple(block) for block in all_blocks]
    tuple_to_idx = {bt: i for i, bt in enumerate(block_tuples)}

    def greedy_cover(random_tiebreak=False, weight_power=1.0, rng=None):
        if rng is None:
            rng = random.Random(42)
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        w = triple_weights ** weight_power if weight_power != 1.0 else triple_weights

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = np.sum(uncovered[block_triples].astype(np.float64) * w[block_triples], axis=1)

            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                best_idx = candidates[rng.randrange(len(candidates))]
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break

            selected.append(int(best_idx))
            uncovered[block_triples[best_idx]] = False

        return selected

    def remove_redundant(block_indices):
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

    def local_search(block_indices, max_iterations=100):
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
                block_triples_bi = block_triples[bi]

                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break

                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue

                critical_triples = block_triples_bi[critical_mask]
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                other_critical = critical_triples[critical_triples != min_ct]

                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(int(cand))
                    else:
                        cand_set = block_sets[cand]
                        all_covered = True
                        for ct in other_critical:
                            if triples[ct] not in cand_set:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(int(cand))

                if valid_candidates:
                    best_cand = valid_candidates[0]
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break

            if not improved:
                break
        return current

    def local_search_remove_two(block_indices, max_iterations=30):
        if len(block_indices) < 2:
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
                for j in range(i + 1, len(current)):
                    bi1 = current[i]
                    bi2 = current[j]
                    both_triples = np.union1d(block_triples[bi1], block_triples[bi2])
                    if np.all(coverage[both_triples] > 2):
                        coverage[both_triples] -= 2
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break
        return current

    def simulated_annealing(block_indices, time_budget=60.0):
        """SA refinement with removal and swap moves."""
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)
        best = list(current)
        best_size = len(current)

        rng = random.Random(777)
        T = 30.0
        T_min = 0.5
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 500

        sa_start = time.time()
        while time.time() - sa_start < time_budget and time.time() < deadline - 5.0:
            T *= cooling
            if T < T_min:
                T = T_min

            # Try removal
            removed = False
            if len(current) > 0:
                indices = list(range(len(current)))
                rng.shuffle(indices)
                for i in indices:
                    bi = current[i]
                    if np.all(coverage[block_triples[bi]] > 1):
                        coverage[block_triples[bi]] -= 1
                        current.pop(i)
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                        removed = True
                        break

            if removed:
                continue

            # Try swap
            if len(current) > 0:
                i = rng.randrange(len(current))
                old_bi = current[i]
                old_covered = block_triples[old_bi]

                critical = old_covered[coverage[old_covered] == 1]
                if len(critical) == 0:
                    # Can just remove
                    coverage[old_covered] -= 1
                    current.pop(i)
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                        no_improve = 0
                    continue

                points_needed = set()
                for ct in critical:
                    points_needed.update(triples[ct])

                if len(points_needed) > k:
                    no_improve += 1
                    continue

                # Find candidates that cover all critical triples
                min_ct = critical[np.argmin(triple_counts[critical])]
                candidates = triple_to_blocks[min_ct]
                other_critical = critical[critical != min_ct]

                valid_candidates = []
                for cand in candidates:
                    cand = int(cand)
                    if cand == old_bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        cand_set = block_sets[cand]
                        all_covered = True
                        for ct in other_critical:
                            if triples[ct] not in cand_set:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(cand)

                if not valid_candidates:
                    no_improve += 1
                    continue

                new_bi = valid_candidates[rng.randrange(len(valid_candidates))]
                new_covered = block_triples[new_bi]

                # Check validity: no triple loses its last cover
                valid = True
                for idx in old_covered:
                    if coverage[idx] == 1 and idx not in new_covered:
                        valid = False
                        break

                if valid:
                    # Compute "gain" for acceptance
                    old_extra = np.sum(coverage[old_covered] > 1)
                    new_extra = np.sum(coverage[new_covered] > 1)
                    delta = new_extra - old_extra  # positive is better

                    if delta >= 0 or rng.random() < np.exp(-delta / T):
                        coverage[old_covered] -= 1
                        coverage[new_covered] += 1
                        current[i] = new_bi
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            if no_improve > max_no_improve:
                break

        # Restore best
        # Rebuild coverage from best
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in best:
            coverage[block_triples[bi]] += 1
        return best

    # Phase 1: Deterministic greedy with different weights
    best_solution = None
    best_size = float('inf')

    for wp in [1.0, 1.5, 2.0]:
        if time.time() > deadline - 30.0:
            break
        selected = greedy_cover(random_tiebreak=False, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=30)
        selected = local_search_remove_two(selected, max_iterations=10)

        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]

    # Phase 2: Random restarts with SA refinement
    np.random.seed(42)
    random.seed(42)

    elapsed = time.time() - start_time
    remaining = deadline - 5.0 - elapsed
    if remaining > 5.0:
        num_restarts = max(1, int(remaining / 3.0))
        num_restarts = min(num_restarts, 30)

        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                break

            wp = random.choice([1.0, 1.5, 2.0, 2.5])
            selected = greedy_cover(random_tiebreak=True, weight_power=wp, rng=random.Random(42 + r))
            selected = remove_redundant(selected)
            selected = local_search(selected, max_iterations=15)

            # SA refinement with time budget
            sa_time = min(15.0, max(3.0, (deadline - 10.0 - time.time()) * 0.3))
            selected = simulated_annealing(selected, time_budget=sa_time)

            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = [list(all_blocks[i]) for i in selected]

    # Phase 3: Final polish on best solution
    if best_solution is not None:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        best_indices = local_search(best_indices, max_iterations=50)
        best_indices = remove_redundant(best_indices)
        best_indices = local_search_remove_two(best_indices, max_iterations=10)

        # Final SA polish
        remaining = deadline - 5.0 - time.time()
        if remaining > 3.0:
            best_indices = simulated_annealing(best_indices, time_budget=min(20.0, remaining - 2.0))

        best_solution = [list(all_blocks[i]) for i in best_indices]
        best_size = len(best_solution)

    return best_solution
