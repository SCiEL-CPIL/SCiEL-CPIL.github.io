import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute covered_indices for all blocks
    all_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]
    
    # Build reverse index: for each tsub, which blocks cover it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for idx in all_covered[i]:
            tsub_blocks[idx].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    def greedy_construct(seed_offset=0):
        """Vectorized greedy construction with optional randomization."""
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []
        local_rng = random.Random(42 + seed_offset * 7919)
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            gains = M @ uncovered
            if gains.max() == 0:
                break
            
            # Add small random perturbation to break ties
            if seed_offset > 0 and gains.max() > 0:
                noise = local_rng.random() * 0.5
                gains = gains + noise * (gains > 0).astype(float)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            covered_indices = all_covered[best_idx]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1

        return selected_indices, counts, selected_mask

    def remove_redundant(indices, counts, mask):
        """Remove all redundant blocks. Returns new indices, counts, mask."""
        changed = True
        while changed:
            changed = False
            for b_idx in indices[:]:
                covered_indices = all_covered[b_idx]
                if np.all(counts[covered_indices] > 1):
                    mask[b_idx] = False
                    indices.remove(b_idx)
                    counts[covered_indices] -= 1
                    changed = True
        return indices, counts, mask

    def try_swaps(indices, counts, mask, max_candidates=30):
        """Try to find a beneficial swap. Returns (improved, indices, counts, mask)."""
        idx_list = indices[:]
        rng.shuffle(idx_list)

        for b_idx in idx_list:
            if time.monotonic() > deadline - 3.0:
                break
            covered_indices = all_covered[b_idx]
            critical_mask = counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]

            if len(critical_indices) == 0:
                # Redundant block
                mask[b_idx] = False
                indices.remove(b_idx)
                counts[covered_indices] -= 1
                return True, indices, counts, mask

            if len(critical_indices) > 10:
                continue  # Too many constraints for a single block

            # Find candidate blocks that cover all critical tsubs
            if len(critical_indices) == 1:
                crit = critical_indices[0]
                candidates = tsub_blocks[crit]
            else:
                # Intersect the block lists for each critical tsub
                common = set(tsub_blocks[critical_indices[0]].tolist())
                for crit in critical_indices[1:]:
                    common &= set(tsub_blocks[crit].tolist())
                    if not common:
                        break
                if not common:
                    continue
                candidates = np.array(list(common), dtype=np.int64)

            # Filter out already-selected blocks
            candidates = candidates[~mask[candidates]]
            if len(candidates) == 0:
                continue

            # Score candidates by how many currently-critical tsubs they cover
            best_cand = None
            best_score = -1
            for cand in candidates[:max_candidates]:
                cand_row = M[cand]
                score = np.sum(cand_row * (counts == 1))
                if score > best_score:
                    best_score = score
                    best_cand = cand

            if best_cand is not None:
                # Perform the swap
                mask[b_idx] = False
                indices.remove(b_idx)
                counts[covered_indices] -= 1
                mask[best_cand] = True
                indices.append(best_cand)
                cand_covered = all_covered[best_cand]
                counts[cand_covered] += 1
                return True, indices, counts, mask

        return False, indices, counts, mask

    def perturb(indices, counts, mask):
        """Perturb the solution by removing some blocks and greedily repairing."""
        unique_mask = (counts == 1).astype(np.uint8)
        gains = M @ unique_mask

        n_remove = min(3 + rng.randint(0, 4), max(1, len(indices) // 4))

        # Select blocks to remove: prefer low-gain blocks
        sorted_indices = np.argsort(gains)
        low_gain_candidates = [i for i in sorted_indices[:max(1, n_remove * 3)] if mask[i]]

        to_remove = set()
        if low_gain_candidates:
            pick_low = rng.sample(low_gain_candidates, min(len(low_gain_candidates), n_remove // 2 + 1))
            to_remove.update(pick_low)

        remaining = [i for i in indices if i not in to_remove]
        if remaining and len(to_remove) < n_remove:
            pick_rand = rng.sample(remaining, min(len(remaining), n_remove - len(to_remove)))
            to_remove.update(pick_rand)

        for b_idx in to_remove:
            if b_idx in indices:
                mask[b_idx] = False
                indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                counts[covered_indices] -= 1

        # Greedily repair
        uncovered_mask = (counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            mask[best_idx] = True
            indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            counts[covered_indices] += 1
            uncovered_mask = (counts == 0).astype(np.uint8)

    def final_polish(indices, counts, mask):
        """High-precision final polish: try all single removals and 1-for-2 swaps."""
        changed = True
        while changed:
            changed = False
            # Try removing each block
            for b_idx in indices[:]:
                if time.monotonic() > deadline - 1.0:
                    break
                covered_indices = all_covered[b_idx]
                if np.all(counts[covered_indices] > 1):
                    mask[b_idx] = False
                    indices.remove(b_idx)
                    counts[covered_indices] -= 1
                    changed = True

            # Try 1-for-2 swaps: remove one block, add two others that together cover its critical tsubs
            if not changed:
                idx_list = indices[:]
                rng.shuffle(idx_list)
                for b_idx in idx_list:
                    if time.monotonic() > deadline - 1.0:
                        break
                    covered_indices = all_covered[b_idx]
                    critical_mask = counts == 1
                    critical_indices = covered_indices[critical_mask[covered_indices]]
                    
                    if len(critical_indices) == 0:
                        continue
                    if len(critical_indices) > 6:
                        continue

                    # Find pairs of blocks that together cover all critical tsubs
                    found_pair = False
                    crit_set = set(critical_indices.tolist())
                    
                    # Get all blocks that cover at least one critical tsub
                    candidate_blocks = set()
                    for crit in critical_indices:
                        candidate_blocks.update(tsub_blocks[crit].tolist())
                    candidate_blocks -= set(indices)
                    
                    if len(candidate_blocks) < 2:
                        continue
                    
                    cand_list = list(candidate_blocks)[:20]
                    for i, c1 in enumerate(cand_list):
                        if time.monotonic() > deadline - 1.0:
                            break
                        # Check which critical tsubs c1 covers
                        c1_covered = set(all_covered[c1].tolist())
                        remaining_crit = crit_set - c1_covered
                        if not remaining_crit:
                            # c1 alone covers all critical - this is a 1-for-1 swap
                            mask[b_idx] = False
                            indices.remove(b_idx)
                            counts[covered_indices] -= 1
                            mask[c1] = True
                            indices.append(c1)
                            counts[all_covered[c1]] += 1
                            changed = True
                            found_pair = True
                            break
                        if len(remaining_crit) > 3:
                            continue
                        for c2 in cand_list[i+1:]:
                            if time.monotonic() > deadline - 1.0:
                                break
                            c2_covered = set(all_covered[c2].tolist())
                            if remaining_crit <= c2_covered:
                                # c1 and c2 together cover all critical
                                mask[b_idx] = False
                                indices.remove(b_idx)
                                counts[covered_indices] -= 1
                                mask[c1] = True
                                indices.append(c1)
                                counts[all_covered[c1]] += 1
                                mask[c2] = True
                                indices.append(c2)
                                counts[all_covered[c2]] += 1
                                changed = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if found_pair:
                        break

        return indices, counts, mask

    # Phase 1: Multiple greedy constructions
    best_indices = None
    best_counts = None
    best_mask = None
    best_len = float('inf')

    for seed_offset in range(5):
        if time.monotonic() > deadline - 20.0:
            break
        sel, cnt, msk = greedy_construct(seed_offset)
        if sel:
            sel, cnt, msk = remove_redundant(sel[:], cnt.copy(), msk.copy())
            if len(sel) < best_len:
                best_len = len(sel)
                best_indices = sel[:]
                best_counts = cnt.copy()
                best_mask = msk.copy()

    if not best_indices:
        # Fallback: pure greedy
        sel, cnt, msk = greedy_construct(0)
        return get_blocks(sel)

    # Phase 2: Simulated Annealing with local search
    current_indices = best_indices[:]
    current_counts = best_counts.copy()
    current_mask = best_mask.copy()

    T = 3.0
    cooling = 0.97
    min_T = 0.05
    stagnation = 0
    max_stagnation = 15

    while time.monotonic() < deadline - 10.0:
        improved = False
        old_len = len(current_indices)

        # Try removing redundant
        current_indices, current_counts, current_mask = remove_redundant(
            current_indices[:], current_counts.copy(), current_mask.copy()
        )
        if len(current_indices) < old_len:
            improved = True

        # Try swaps
        if not improved:
            improved, current_indices, current_counts, current_mask = try_swaps(
                current_indices[:], current_counts.copy(), current_mask.copy()
            )

        if improved:
            stagnation = 0
            T = max(T * cooling, min_T)
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_indices = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            if stagnation >= max_stagnation or (T > min_T and rng.random() < 0.3):
                # Save state
                save_indices = current_indices[:]
                save_counts = current_counts.copy()
                save_mask = current_mask.copy()
                save_len = len(current_indices)

                perturb(current_indices, current_counts, current_mask)
                current_indices, current_counts, current_mask = remove_redundant(
                    current_indices[:], current_counts.copy(), current_mask.copy()
                )

                new_len = len(current_indices)
                delta = new_len - save_len

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = True
                else:
                    p_accept = np.exp(-delta / max(T, 0.1))
                    if rng.random() < p_accept:
                        accept = True

                if accept:
                    T = max(T * cooling, min_T)
                    stagnation = 0
                    if new_len < best_len:
                        best_len = new_len
                        best_indices = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                else:
                    current_indices = save_indices
                    current_counts = save_counts
                    current_mask = save_mask
                    T = min(T * 1.05, 8.0)
                    stagnation = 0

            T = max(T * cooling, min_T)

    # Phase 3: Final high-precision polish
    final_indices = best_indices[:]
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    final_indices, final_counts, final_mask = final_polish(
        final_indices, final_counts, final_mask
    )

    # Verify validity
    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        verify_counts[all_covered[b_idx]] += 1

    if not np.all(verify_counts >= 1):
        # Fallback to greedy solution
        sel, cnt, msk = greedy_construct(0)
        return get_blocks(sel)

    return get_blocks(final_indices)
