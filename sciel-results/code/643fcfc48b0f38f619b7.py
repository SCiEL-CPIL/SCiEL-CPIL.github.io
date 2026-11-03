import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0
    deadline = start_time + limit

    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}

    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    block_covers = []
    sub_to_blocks = [[] for _ in range(n_t)]

    for i, block in enumerate(blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))

    if not _deadline_ok(deadline):
        return []

    # 2. Fast Greedy Initialization (multiple diverse starts)
    def greedy_init(rng, deadline):
        uncovered_mask = np.ones(n_t, dtype=bool)
        gains = np.full(n_blocks, 0, dtype=np.int32)
        initial_gain = len(block_covers[0]) if n_blocks > 0 else 0
        gains[:] = initial_gain

        selected = []
        selected_set = set()

        while np.any(uncovered_mask):
            if not _deadline_ok(deadline):
                break

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Add small randomization to break ties
            candidates = np.where(gains >= max_gain - rng.randint(0, 2))[0]
            candidates = [c for c in candidates if c not in selected_set]
            if not candidates:
                best_block = np.argmax(gains)
                if best_block in selected_set:
                    gains[best_block] = -1
                    continue
            else:
                best_block = rng.choice(candidates)

            if best_block in selected_set:
                gains[best_block] = -1
                continue

            selected.append(best_block)
            selected_set.add(best_block)

            covers_b = block_covers[best_block]
            for s_idx in covers_b:
                if uncovered_mask[s_idx]:
                    uncovered_mask[s_idx] = False
                    for b_prime in sub_to_blocks[s_idx]:
                        gains[b_prime] -= 1

            gains[best_block] = -1

        return selected

    rng = random.Random(42)

    # Generate multiple diverse initial solutions
    best_selected = None
    best_size = float('inf')
    num_starts = min(5, max(1, int((deadline - time.time()) / 20)))

    for start_i in range(num_starts):
        if not _deadline_ok(deadline):
            break
        selected = greedy_init(rng, deadline)
        if selected and len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected

    if not best_selected:
        best_selected = greedy_init(rng, deadline)

    if not best_selected:
        return []

    # 3. Build counts
    def build_counts(block_list):
        c = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            c[block_covers[b]] += 1
        return c

    counts = build_counts(best_selected)

    def prune_redundant(selected_list, counts, deadline):
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in range(len(selected_list) - 1, -1, -1):
                b = selected_list[i]
                covers_b = block_covers[b]
                if np.all(counts[covers_b] > 1):
                    selected_list.pop(i)
                    counts[covers_b] -= 1
                    changed = True
        return changed

    selected_list = list(best_selected)
    prune_redundant(selected_list, counts, deadline)

    best_size = len(selected_list)
    best_solution = selected_list.copy()
    best_counts = counts.copy()

    all_block_indices = list(range(n_blocks))
    current_set = set(selected_list)

    # Helper: vectorized greedy repair
    def greedy_repair(block_list, counts, deadline):
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            return True

        max_iters = 200
        for _ in range(max_iters):
            if not _deadline_ok(deadline):
                return False

            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                return True

            # Sample uncovered subsets for speed
            sample_size = min(100, len(uncovered))
            sample_idx = rng.sample(uncovered.tolist(), sample_size)

            # Find best block covering most uncovered
            best_block = -1
            best_gain = 0
            for s_idx in sample_idx:
                for b_prime in sub_to_blocks[s_idx]:
                    if b_prime not in current_set:
                        cov = block_covers[b_prime]
                        gain = np.sum(counts[cov] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_block = b_prime

            if best_block == -1 or best_gain == 0:
                break

            block_list.append(best_block)
            current_set.add(best_block)
            counts[block_covers[best_block]] += 1

        return np.all(counts > 0)

    # 4. Local Search with 2-for-1 swaps and ILS

    def try_add_remove(selected_list, counts, current_set, deadline):
        """Try adding one block and removing one to improve."""
        # Find weakly covered subsets
        min_count = np.min(counts)
        if min_count <= 2:
            weak_subs = np.where(counts <= min_count + 1)[0]
            if len(weak_subs) > 0:
                target_sub = rng.choice(weak_subs.tolist())
                candidates = [b for b in sub_to_blocks[target_sub] if b not in current_set]
                if candidates:
                    b_add = rng.choice(candidates)
                else:
                    return None
            else:
                b_add = rng.choice(all_block_indices)
        else:
            b_add = rng.choice(all_block_indices)

        if b_add in current_set:
            return None

        temp_counts = counts.copy()
        temp_counts[block_covers[b_add]] += 1

        # Try removing each block that becomes redundant
        for i in range(len(selected_list) - 1, -1, -1):
            b = selected_list[i]
            covers_b = block_covers[b]
            if np.all(temp_counts[covers_b] > 1):
                # Can remove b
                new_list = selected_list.copy()
                new_list.pop(i)
                new_counts = temp_counts.copy()
                new_counts[covers_b] -= 1
                return (new_list, new_counts, b_add)

        return None

    def try_two_for_one(selected_list, counts, current_set, deadline):
        """Try adding two blocks and removing one."""
        # Find weakly covered subsets
        min_count = np.min(counts)
        weak_subs = np.where(counts <= min_count + 1)[0]

        if len(weak_subs) == 0:
            return None

        target1 = rng.choice(weak_subs.tolist())
        candidates1 = [b for b in sub_to_blocks[target1] if b not in current_set]
        if not candidates1:
            return None
        b_add1 = rng.choice(candidates1)

        # Find another weak subset
        temp_counts = counts.copy()
        temp_counts[block_covers[b_add1]] += 1
        weak_subs2 = np.where(temp_counts <= 2)[0]
        if len(weak_subs2) == 0:
            return None
        target2 = rng.choice(weak_subs2.tolist())
        candidates2 = [b for b in sub_to_blocks[target2] if b not in current_set and b != b_add1]
        if not candidates2:
            return None
        b_add2 = rng.choice(candidates2)

        temp_counts[block_covers[b_add2]] += 1

        # Try removing each block that becomes redundant
        for i in range(len(selected_list) - 1, -1, -1):
            b = selected_list[i]
            covers_b = block_covers[b]
            if np.all(temp_counts[covers_b] > 1):
                new_list = selected_list.copy()
                new_list.pop(i)
                new_counts = temp_counts.copy()
                new_counts[covers_b] -= 1
                return (new_list, new_counts, b_add1, b_add2)

        return None

    perturbation_strength = 3

    while _deadline_ok(deadline):
        # Phase 1: Local search iterations
        for _ in range(30):
            if not _deadline_ok(deadline):
                break

            # Try 1-for-1 swap
            result = try_add_remove(selected_list, counts, current_set, deadline)
            if result:
                new_list, new_counts, b_add = result
                current_set.discard(new_list[0] if len(new_list) > 0 else -1)
                # Rebuild set properly
                current_set = set(new_list)
                current_set.add(b_add)

                prune_redundant(new_list, new_counts, deadline)
                current_size = len(new_list)

                if current_size < best_size:
                    best_size = current_size
                    best_solution = new_list.copy()
                    best_counts = new_counts.copy()
                    selected_list = new_list
                    counts = new_counts
                    current_set = set(selected_list)
                elif current_size < len(selected_list):
                    selected_list = new_list
                    counts = new_counts
                    current_set = set(selected_list)

                if len(selected_list) > best_size + 5:
                    selected_list = best_solution.copy()
                    counts = best_counts.copy()
                    current_set = set(selected_list)
            else:
                # Try 2-for-1 swap
                result2 = try_two_for_one(selected_list, counts, current_set, deadline)
                if result2:
                    new_list, new_counts, b_add1, b_add2 = result2
                    prune_redundant(new_list, new_counts, deadline)
                    current_size = len(new_list)

                    if current_size < best_size:
                        best_size = current_size
                        best_solution = new_list.copy()
                        best_counts = new_counts.copy()
                        selected_list = new_list
                        counts = new_counts
                        current_set = set(selected_list)
                    elif current_size < len(selected_list):
                        selected_list = new_list
                        counts = new_counts
                        current_set = set(selected_list)

                    if len(selected_list) > best_size + 5:
                        selected_list = best_solution.copy()
                        counts = best_counts.copy()
                        current_set = set(selected_list)

        # Phase 2: Perturbation
        if not _deadline_ok(deadline):
            break

        pert_list = best_solution.copy()
        pert_set = set(pert_list)
        pert_counts = best_counts.copy()

        num_remove = min(perturbation_strength, len(pert_list) // 3)
        if num_remove > 0:
            remove_indices = rng.sample(range(len(pert_list)), num_remove)
            for idx in sorted(remove_indices, reverse=True):
                b = pert_list[idx]
                pert_list.pop(idx)
                pert_set.discard(b)
                pert_counts[block_covers[b]] -= 1

        old_current_set = current_set
        current_set = pert_set
        repaired = greedy_repair(pert_list, pert_counts, deadline)
        current_set = old_current_set

        if repaired:
            prune_redundant(pert_list, pert_counts, deadline)
            if len(pert_list) < best_size:
                best_size = len(pert_list)
                best_solution = pert_list.copy()
                best_counts = pert_counts.copy()
                selected_list = pert_list
                current_set = set(best_solution)
                counts = pert_counts

        # Adapt perturbation strength
        if rng.random() < 0.15:
            perturbation_strength = min(perturbation_strength + 1, 15)
        elif rng.random() < 0.25:
            perturbation_strength = max(2, perturbation_strength - 1)

    # Final Result
    result = [list(blocks[i]) for i in best_solution]
    return result
