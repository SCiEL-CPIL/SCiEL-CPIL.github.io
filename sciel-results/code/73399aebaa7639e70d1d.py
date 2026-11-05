import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs) as uint8
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute which tsubs each block covers (for quick access)
    block_tsub_lists = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]

    # Precompute tsub_to_blocks (list of block indices for each tsub)
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx in range(num_kblocks):
        for s in block_tsub_lists[b_idx]:
            tsub_to_blocks[s].append(b_idx)

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Helper: Greedy construction using vectorized gains
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        selected_set = set()

        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            covered_indices = block_tsub_lists[best_idx]
            uncovered[covered_indices] = 0

        return selected_indices, selected_set

    # Helper: Remove redundant blocks using vectorized checks
    def remove_redundant(indices, indices_set):
        if not indices:
            return indices, indices_set
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]

        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in list(indices_set):
                covered = block_tsub_lists[idx]
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)

            for idx in to_remove:
                indices_set.remove(idx)
                indices.remove(idx)
                counts -= M[idx]
                changed = True

        return indices, indices_set

    # Helper: Local search - deletion and swap
    def local_search(indices, indices_set):
        if not indices:
            return indices, indices_set

        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]

        improved = True
        while improved and time.monotonic() < deadline - 3.0:
            improved = False

            # Pass 1: Remove redundant
            indices, indices_set = remove_redundant(indices, indices_set)
            if len(indices) < len(indices_set) or True:  # Always recompute counts after potential removal
                counts = np.zeros(num_tsubs, dtype=np.int32)
                for idx in indices_set:
                    counts += M[idx]
                if len(indices) < len(indices_set):
                    improved = True
                    continue

            # Pass 2: Try swaps
            idx_list = list(indices_set)
            rng.shuffle(idx_list)

            for b_idx in idx_list:
                if time.monotonic() > deadline - 3.0:
                    return indices, indices_set

                covered = block_tsub_lists[b_idx]
                critical_mask = counts == 1
                critical_indices = covered[critical_mask[covered]]

                if len(critical_indices) == 0:
                    # Redundant
                    indices_set.remove(b_idx)
                    indices.remove(b_idx)
                    counts -= M[b_idx]
                    improved = True
                    break

                # Find candidate block that covers all critical tsubs
                if len(critical_indices) > 30:
                    continue  # Too many constraints, unlikely to find single block

                # Use vectorized check: M[:, critical_indices].all(axis=1)
                candidates_mask = M[:, critical_indices].all(axis=1)
                candidates = np.where(candidates_mask)[0]
                candidates = [c for c in candidates if c not in indices_set]

                if len(candidates) > 0:
                    # Pick a candidate that maximizes additional coverage
                    best_c = candidates[0]
                    best_gain = -1
                    for c in candidates[:50]:  # Limit to first 50 for speed
                        gain = np.sum(M[c] * (counts == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c

                    # Perform swap
                    indices_set.remove(b_idx)
                    indices.remove(b_idx)
                    indices_set.add(best_c)
                    indices.append(best_c)
                    counts -= M[b_idx]
                    counts += M[best_c]
                    improved = True
                    break

        return indices, indices_set

    # Helper: Simulated Annealing
    def simulated_annealing(indices, indices_set, t0=10.0, t_end=0.1, cooling=0.995, max_iter=5000):
        if not indices:
            return indices, indices_set

        current_len = len(indices)
        best_indices = indices[:]
        best_set = set(indices_set)
        best_len = current_len

        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]

        temperature = t0
        iter_count = 0

        while temperature > t_end and iter_count < max_iter:
            if time.monotonic() > deadline - 10.0:
                break

            # Generate a move: swap a random block with a random candidate
            if len(indices) < 2:
                break

            # Pick a random block to swap
            i = rng.randrange(len(indices))
            old_block = indices[i]

            # Find critical tsubs for this block
            covered = block_tsub_lists[old_block]
            critical_mask = counts == 1
            critical_indices = covered[critical_mask[covered]]

            # Find candidate blocks that cover all critical tsubs
            if len(critical_indices) == 0:
                # Block is redundant, just remove it (this is a "swap with nothing")
                new_len = current_len - 1
                delta = -1  # Improvement
            else:
                if len(critical_indices) > 25:
                    temperature *= cooling
                    iter_count += 1
                    continue

                candidates_mask = M[:, critical_indices].all(axis=1)
                candidates = np.where(candidates_mask)[0]
                candidates = [c for c in candidates if c not in indices_set]

                if not candidates:
                    temperature *= cooling
                    iter_count += 1
                    continue

                # Pick a random candidate
                new_block = candidates[rng.randrange(len(candidates))]
                new_len = current_len  # Swap keeps same length

            # Compute energy change (for covering design, energy = number of blocks)
            # But we also want to consider coverage quality
            if len(critical_indices) == 0:
                # Removal
                indices_set.remove(old_block)
                indices.remove(old_block)
                counts -= M[old_block]
                current_len = new_len
                delta = -1
            else:
                # Swap
                # Check if new_block adds coverage beyond what old_block provides
                old_only = np.sum(M[old_block] * (counts == 1))
                new_gain = np.sum(M[new_block] * (counts == 0))

                indices_set.remove(old_block)
                indices[i] = new_block
                indices_set.add(new_block)
                counts -= M[old_block]
                counts += M[new_block]
                delta = 0  # Same number of blocks

            # Accept or reject based on temperature
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-10)):
                # Accept
                if current_len < best_len:
                    best_len = current_len
                    best_indices = indices[:]
                    best_set = set(indices_set)
            else:
                # Reject: revert
                if len(critical_indices) == 0:
                    indices_set.add(old_block)
                    indices.append(old_block)
                    counts += M[old_block]
                    current_len += 1
                else:
                    indices_set.remove(new_block)
                    indices[i] = old_block
                    indices_set.add(old_block)
                    counts -= M[new_block]
                    counts += M[old_block]

            temperature *= cooling
            iter_count += 1

        # Final local search polish
        best_indices, best_set = local_search(best_indices, best_set)

        return best_indices, best_set

    # Helper: Perturbation for ILS
    def perturb(indices, indices_set, num_remove=3):
        if len(indices) <= num_remove:
            return greedy_construction()

        indices_to_remove = rng.sample(range(len(indices)), num_remove)
        indices_to_remove.sort(reverse=True)

        new_indices = indices[:]
        new_set = set(indices_set)
        for idx in indices_to_remove:
            b_idx = new_indices[idx]
            new_set.remove(b_idx)
            new_indices.pop(idx)

        # Repair: add blocks to cover uncovered tsubs
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in new_set:
            counts += M[idx]

        uncovered_mask = (counts == 0)
        uncovered_indices = np.where(uncovered_mask)[0]

        while len(uncovered_indices) > 0 and time.monotonic() < deadline - 5.0:
            # Pick an uncovered tsub and find a block covering it
            target = uncovered_indices[rng.randrange(len(uncovered_indices))]
            candidates = tsub_to_blocks[target]
            candidates = [c for c in candidates if c not in new_set]

            if not candidates:
                break

            # Pick candidate that covers most uncovered
            best_c = candidates[0]
            best_cov = 0
            for c in candidates[:30]:
                cov = np.sum(M[c] * uncovered_mask)
                if cov > best_cov:
                    best_cov = cov
                    best_c = c

            new_set.add(best_c)
            new_indices.append(best_c)
            counts += M[best_c]
            newly_covered = block_tsub_lists[best_c]
            uncovered_mask[newly_covered] = (counts[newly_covered] == 0)
            uncovered_indices = np.where(uncovered_mask)[0]

        return new_indices, new_set

    # Main algorithm
    # Phase 1: Greedy initialization
    current_indices, current_set = greedy_construction()
    best_indices, best_set = local_search(current_indices, current_set)
    best_len = len(best_indices)

    # Phase 2: Simulated Annealing
    sa_indices, sa_set = simulated_annealing(current_indices, current_set)
    if len(sa_indices) < best_len:
        best_len = len(sa_indices)
        best_indices = sa_indices[:]
        best_set = sa_set

    # Phase 3: ILS with SA as local search
    current_indices, current_set = best_indices[:], set(best_set)

    while time.monotonic() < deadline - 15.0:
        # Perturb
        pert_indices, pert_set = perturb(current_indices, current_set, rng.randint(2, 5))

        # Local search with SA
        opt_indices, opt_set = local_search(pert_indices, pert_set)
        if len(opt_indices) == len(current_indices):
            # Try SA for additional improvement
            sa_idx, sa_set2 = simulated_annealing(opt_indices, opt_set, t0=5.0, t_end=0.2, cooling=0.99, max_iter=1000)
            if len(sa_idx) < len(opt_indices):
                opt_indices, opt_set = sa_idx, sa_set2

        # Update best
        if len(opt_indices) < best_len:
            best_len = len(opt_indices)
            best_indices = opt_indices[:]
            best_set = opt_set

        # Accept if better than current or with some probability
        if len(opt_indices) <= len(current_indices):
            current_indices = opt_indices[:]
            current_set = opt_set
        elif rng.random() < 0.1:
            # Occasionally accept worse to explore
            current_indices = opt_indices[:]
            current_set = opt_set

    # Final polish
    best_indices, best_set = local_search(best_indices, best_set)

    # Convert to final blocks
    final_blocks = [list(all_kblocks[b_idx]) for b_idx in best_indices]
    return final_blocks
