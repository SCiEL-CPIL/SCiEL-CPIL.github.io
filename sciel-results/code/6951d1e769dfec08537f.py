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

    # Precompute covered indices for each block
    block_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]
    
    # Reverse index: tsub -> blocks
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for j in block_covered[i]:
            tsub_blocks[j].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    def greedy_construct(seed):
        rng_local = random.Random(seed)
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []

        while np.any(uncovered):
            if time.monotonic() > deadline - 10.0:
                break
            gains = M @ uncovered
            # Add randomness to tie-breaking
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            # Randomly pick one
            best_idx = int(rng_local.choice(candidates))
            
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            covered_indices = block_covered[best_idx]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1

        return selected_indices, counts, selected_mask

    def local_search(initial_indices, initial_counts, initial_mask):
        current_indices = initial_indices[:]
        current_counts = initial_counts.copy()
        current_mask = initial_mask.copy()
        
        best_len = len(current_indices)
        best_solution = current_indices[:]
        best_counts = current_counts.copy()
        best_mask = current_mask.copy()

        def try_remove_redundant():
            removed = False
            changed = True
            while changed:
                changed = False
                # Shuffle to avoid bias
                idx_list = current_indices[:]
                rng.shuffle(idx_list)
                for b_idx in idx_list:
                    covered_indices = block_covered[b_idx]
                    if np.all(current_counts[covered_indices] > 1):
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        current_counts[covered_indices] -= 1
                        removed = True
                        changed = True
            return removed

        def try_swaps():
            improved = False
            idx_list = current_indices[:]
            rng.shuffle(idx_list)

            for b_idx in idx_list:
                if time.monotonic() > deadline - 2.0:
                    break
                covered_indices = block_covered[b_idx]
                critical_mask = current_counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]

                if len(critical_indices) == 0:
                    # Redundant
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    improved = True
                    break

                if len(critical_indices) <= 3:
                    # Try to find a block that covers all criticals
                    common = None
                    for crit in critical_indices:
                        crit = int(crit)
                        blocks_c = tsub_blocks[crit]
                        if common is None:
                            common = set(blocks_c.tolist())
                        else:
                            common &= set(blocks_c.tolist())
                    
                    if common:
                        candidates = np.array(list(common), dtype=np.int64)
                        candidates = candidates[~current_mask[candidates]]
                        if len(candidates) > 0:
                            # Pick candidate that covers most other criticals or just any
                            best_cand = int(rng.choice(candidates))
                            current_mask[b_idx] = False
                            current_indices.remove(b_idx)
                            current_counts[covered_indices] -= 1
                            current_mask[best_cand] = True
                            current_indices.append(best_cand)
                            cand_covered = block_covered[best_cand]
                            current_counts[cand_covered] += 1
                            improved = True
                            break
            return improved

        def perturb():
            n_remove = min(3 + rng.randint(0, 3), max(1, len(current_indices) // 4))
            to_remove = rng.sample(current_indices, n_remove)
            for b_idx in to_remove:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = block_covered[b_idx]
                current_counts[covered_indices] -= 1

            uncovered_mask = (current_counts == 0).astype(np.uint8)
            while np.any(uncovered_mask):
                if time.monotonic() > deadline - 5.0:
                    break
                gains = M @ uncovered_mask
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                current_mask[best_idx] = True
                current_indices.append(best_idx)
                covered_indices = block_covered[best_idx]
                current_counts[covered_indices] += 1
                uncovered_mask = (current_counts == 0).astype(np.uint8)

        stagnation = 0
        max_stagnation = 20

        while time.monotonic() < deadline - 3.0:
            improved = False

            if try_remove_redundant():
                improved = True

            if not improved and try_swaps():
                improved = True

            if improved:
                stagnation = 0
                if len(current_indices) < best_len:
                    best_len = len(current_indices)
                    best_solution = current_indices[:]
                    best_counts = current_counts.copy()
                    best_mask = current_mask.copy()
            else:
                stagnation += 1
                if stagnation >= max_stagnation:
                    perturb()
                    try_remove_redundant()
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_solution = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                    stagnation = 0

        return best_solution

    # Run multiple greedy starts
    best_final_indices = None
    min_len = float('inf')
    
    num_starts = 5
    for i in range(num_starts):
        if time.monotonic() > deadline - 20.0:
            break
        idx, counts, mask = greedy_construct(42 + i)
        if not idx:
            continue
        
        # Local search
        final_idx = local_search(idx, counts, mask)
        
        if len(final_idx) < min_len:
            min_len = len(final_idx)
            best_final_indices = final_idx
            
    if best_final_indices is None:
        # Fallback
        idx, counts, mask = greedy_construct(42)
        best_final_indices = idx

    # Final verification and cleanup
    final_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in best_final_indices:
        covered_indices = block_covered[b_idx]
        final_counts[covered_indices] += 1

    if not np.all(final_counts >= 1):
        # Fallback to greedy only
        idx, counts, mask = greedy_construct(42)
        best_final_indices = idx

    return get_blocks(best_final_indices)
