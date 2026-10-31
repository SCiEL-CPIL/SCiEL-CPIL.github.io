import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(indices)

    # Reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Build numpy array of variable-length arrays
    block_to_tsubs_np = np.array(block_to_tsubs, dtype=np.int32)

    # Precompute a 2D coverage matrix: cov_matrix[block_idx, t_idx] = 1 if block covers t
    cov_matrix = np.zeros((num_blocks, num_t_subsets), dtype=np.int8)
    for b_idx, t_indices in enumerate(block_to_tsubs):
        cov_matrix[b_idx, t_indices] = 1

    # Precompute block bitsets for fast intersection
    block_bitsets = []
    for t_indices in block_to_tsubs:
        bits = 0
        for idx in t_indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)

    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    blocks_in_cover = []
    in_cover_set = set()

    def add_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] += 1
        blocks_in_cover.append(b_idx)
        in_cover_set.add(b_idx)

    def remove_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] -= 1
        in_cover_set.discard(b_idx)
        if b_idx in blocks_in_cover:
            blocks_in_cover.remove(b_idx)

    def find_redundant_blocks():
        """Vectorized: find all blocks that are redundant (can be removed)."""
        if not blocks_in_cover:
            return []
        redundant = []
        for b_idx in blocks_in_cover:
            t_indices = block_to_tsubs_np[b_idx]
            if np.all(cover_count[t_indices] > 1):
                redundant.append(b_idx)
        return redundant

    def get_unique_tsubs(b_idx):
        """Get t-subsets only covered by b_idx."""
        t_indices = block_to_tsubs_np[b_idx]
        return t_indices[cover_count[t_indices] == 1]

    # 1. Greedy Initialization with full constrained-first ordering
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    num_uncovered = num_t_subsets

    # Precompute candidate counts for each t-subset
    t_sub_cand_counts = np.array([len(t_sub_to_blocks[i]) for i in range(num_t_subsets)], dtype=np.int32)

    while num_uncovered > 0:
        if time.time() > deadline - 25:
            break

        # Find the most constrained uncovered t-subset
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        # Sort by constraint (fewest candidates first), take top few
        if len(uncovered_indices) > 20:
            constrained = uncovered_indices[np.argsort(t_sub_cand_counts[uncovered_indices])][:20]
        else:
            constrained = uncovered_indices

        best_block_idx = None
        max_gain = 0

        for target_idx in constrained:
            candidates = t_sub_to_blocks[target_idx]
            cand_arr = np.array(candidates, dtype=np.int32)
            # Vectorized gain: count newly covered t-subsets
            gains = np.sum(cov_matrix[cand_arr] & (cover_count == 0), axis=1)
            best_local = np.argmax(gains)
            local_gain = gains[best_local]
            if local_gain > max_gain:
                max_gain = local_gain
                best_block_idx = cand_arr[best_local]

        if best_block_idx is not None and max_gain > 0:
            add_block(best_block_idx)
            newly_covered = block_to_tsubs_np[best_block_idx]
            was_uncovered = uncovered_mask[newly_covered]
            now_covered = cover_count[newly_covered] > 0
            num_uncovered -= np.sum(was_uncovered & now_covered)
            uncovered_mask[newly_covered] = ~now_covered
        else:
            break

    # 2. Aggressive Local Search
    best_size = len(blocks_in_cover)
    best_blocks_snapshot = list(blocks_in_cover)

    no_improvement_count = 0
    max_no_improvement = 30
    restart_count = 0
    max_restarts = 8

    while time.time() < deadline - 10:
        improved = False

        # Phase 1: Delete all redundant blocks
        to_remove = find_redundant_blocks()
        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True

        if improved:
            no_improvement_count = 0
            if len(blocks_in_cover) < best_size:
                best_size = len(blocks_in_cover)
                best_blocks_snapshot = list(blocks_in_cover)
            continue

        # Phase 2: Try 1-for-1 swaps for critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            t_indices = block_to_tsubs_np[b_idx]
            if np.any(cover_count[t_indices] == 1):
                critical_blocks.append(b_idx)

        if critical_blocks:
            random.shuffle(critical_blocks)
            swapped = False

            for b_old_idx in critical_blocks:
                if time.time() > deadline - 10:
                    break

                unique_tsubs = get_unique_tsubs(b_old_idx)
                if len(unique_tsubs) == 0:
                    continue

                # Use bitset intersection to find candidates covering all unique t-subsets
                mask = block_bitsets[unique_tsubs[0]]
                for t_idx in unique_tsubs[1:]:
                    mask &= block_bitsets[t_idx]
                    if mask == 0:
                        break

                if mask == 0:
                    continue

                # Find all blocks whose bitset is a superset of the required mask
                # i.e., block covers all unique t-subsets
                candidates = []
                for b_new_idx in range(num_blocks):
                    if b_new_idx == b_old_idx:
                        continue
                    if (block_bitsets[b_new_idx] & mask) == mask:
                        candidates.append(b_new_idx)

                if not candidates:
                    continue

                # Pick candidate that maximizes additional coverage
                best_new = None
                best_score = -1
                random.shuffle(candidates)
                for b_new_idx in candidates[:40]:
                    score = np.sum(cov_matrix[b_new_idx] & (cover_count == 0))
                    if score > best_score:
                        best_score = score
                        best_new = b_new_idx

                if best_new is not None:
                    remove_block(b_old_idx)
                    add_block(best_new)
                    improved = True
                    swapped = True
                    break

            if swapped:
                no_improvement_count = 0
                if len(blocks_in_cover) < best_size:
                    best_size = len(blocks_in_cover)
                    best_blocks_snapshot = list(blocks_in_cover)
                continue

        # Phase 3: Try 2-for-1 swaps
        if len(blocks_in_cover) > 10 and time.time() < deadline - 15:
            two_swap_done = False
            n_crit = len(critical_blocks) if critical_blocks else 0

            for i in range(min(15, n_crit)):
                if time.time() > deadline - 15:
                    break
                if not critical_blocks:
                    break

                b1 = critical_blocks[i % n_crit]
                if b1 not in in_cover_set:
                    continue

                u1 = get_unique_tsubs(b1)
                if len(u1) == 0:
                    continue

                for j in range(i + 1, min(i + 8, n_crit)):
                    b2 = critical_blocks[j % n_crit]
                    if b2 == b1 or b2 not in in_cover_set:
                        continue
                    u2 = get_unique_tsubs(b2)
                    if len(u2) == 0:
                        continue

                    combined = np.union1d(u1, u2)

                    # Bitset intersection
                    mask = block_bitsets[combined[0]]
                    for t_idx in combined[1:]:
                        mask &= block_bitsets[t_idx]
                        if mask == 0:
                            break

                    if mask == 0:
                        continue

                    candidates = []
                    for b_new_idx in range(num_blocks):
                        if b_new_idx == b1 or b_new_idx == b2:
                            continue
                        if (block_bitsets[b_new_idx] & mask) == mask:
                            candidates.append(b_new_idx)

                    if not candidates:
                        continue

                    best_new = None
                    best_score = -1
                    random.shuffle(candidates)
                    for b_new_idx in candidates[:20]:
                        score = np.sum(cov_matrix[b_new_idx] & (cover_count == 0))
                        if score > best_score:
                            best_score = score
                            best_new = b_new_idx

                    if best_new is not None:
                        remove_block(b1)
                        remove_block(b2)
                        add_block(best_new)
                        improved = True
                        two_swap_done = True
                        break

                if two_swap_done:
                    break

            if two_swap_done:
                no_improvement_count = 0
                if len(blocks_in_cover) < best_size:
                    best_size = len(blocks_in_cover)
                    best_blocks_snapshot = list(blocks_in_cover)
                continue

        # No improvement - consider restart
        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            restart_count += 1
            if restart_count > max_restarts:
                break

            # Restore best snapshot
            blocks_in_cover.clear()
            in_cover_set.clear()
            cover_count.fill(0)
            for b_idx in best_blocks_snapshot:
                add_block(b_idx)

            # Perturb: remove some random blocks and re-greedy
            if len(blocks_in_cover) > 5:
                num_remove = max(1, len(blocks_in_cover) // 8)
                to_perturb = random.sample(blocks_in_cover, min(num_remove, len(blocks_in_cover)))
                for b_idx in to_perturb:
                    remove_block(b_idx)

                # Re-greedy to fill gaps
                uncovered_mask = (cover_count == 0)
                while np.any(uncovered_mask) and time.time() < deadline - 10:
                    uncovered_indices = np.where(uncovered_mask)[0]
                    if len(uncovered_indices) == 0:
                        break
                    # Pick most constrained
                    if len(uncovered_indices) > 15:
                        constrained = uncovered_indices[np.argsort(t_sub_cand_counts[uncovered_indices])][:15]
                    else:
                        constrained = uncovered_indices

                    best_block_idx = None
                    max_gain = 0
                    for target_idx in constrained:
                        candidates = t_sub_to_blocks[target_idx]
                        cand_arr = np.array(candidates, dtype=np.int32)
                        gains = np.sum(cov_matrix[cand_arr] & (cover_count == 0), axis=1)
                        best_local = np.argmax(gains)
                        local_gain = gains[best_local]
                        if local_gain > max_gain:
                            max_gain = local_gain
                            best_block_idx = cand_arr[best_local]

                    if best_block_idx is not None and max_gain > 0:
                        add_block(best_block_idx)
                        newly_covered = block_to_tsubs_np[best_block_idx]
                        uncovered_mask[newly_covered] = (cover_count[newly_covered] == 0)
                    else:
                        break

            no_improvement_count = 0

    # Final polish: exhaustive redundant removal + one more round of swaps
    changed = True
    while changed:
        changed = False
        to_remove = find_redundant_blocks()
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True

    # One final pass of 1-for-1 swaps
    for b_old_idx in list(blocks_in_cover):
        if time.time() > deadline - 2:
            break
        unique_tsubs = get_unique_tsubs(b_old_idx)
        if len(unique_tsubs) == 0:
            continue
        mask = block_bitsets[unique_tsubs[0]]
        for t_idx in unique_tsubs[1:]:
            mask &= block_bitsets[t_idx]
            if mask == 0:
                break
        if mask == 0:
            continue
        best_new = None
        best_score = -1
        for b_new_idx in range(num_blocks):
            if b_new_idx == b_old_idx:
                continue
            if (block_bitsets[b_new_idx] & mask) == mask:
                score = np.sum(cov_matrix[b_new_idx] & (cover_count == 0))
                if score > best_score:
                    best_score = score
                    best_new = b_new_idx
        if best_new is not None:
            remove_block(b_old_idx)
            add_block(best_new)

    # Final redundant cleanup
    changed = True
    while changed:
        changed = False
        to_remove = find_redundant_blocks()
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True

    # Use best if current is worse
    if len(blocks_in_cover) > best_size:
        blocks_in_cover = list(best_blocks_snapshot)

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    final_blocks.sort()

    return final_blocks
