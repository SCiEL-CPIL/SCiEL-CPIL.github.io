import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 132.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Step 3: For each block, compute which t-subsets it covers
    block_t_subsets = []
    block_bits = []
    for block in k_blocks:
        covered = set()
        bits = 0
        for combo in itertools.combinations(block, t):
            idx = t_index[combo]
            covered.add(idx)
            bits |= (1 << idx)
        block_t_subsets.append(covered)
        block_bits.append(bits)

    if elapsed() > deadline - 40:
        return [list(b) for b in k_blocks]

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)

    # Step 4: Weighted greedy set cover with vectorized gain updates
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    t_weights = 1.0 / safe_counts

    # Precompute block coverage as numpy arrays for fast set operations
    block_cover_arrays = []
    for covered in block_t_subsets:
        if covered:
            block_cover_arrays.append(np.fromiter(covered, dtype=np.int32, count=len(covered)))
        else:
            block_cover_arrays.append(np.array([], dtype=np.int32))

    gain = np.zeros(num_k, dtype=np.float64)
    for i, arr in enumerate(block_cover_arrays):
        if arr.size > 0:
            gain[i] = np.sum(t_weights[arr])

    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)

    while uncovered_count > 0 and elapsed() < deadline - 30:
        masked_gain = gain.copy()
        masked_gain[selected_mask] = -1.0
        best_idx = int(np.argmax(masked_gain))
        best_gain = gain[best_idx]

        if best_gain <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

        # Vectorized update: for each newly uncovered t-subset, subtract its weight from all covering blocks
        new_covered = block_cover_arrays[best_idx]
        newly_uncovered = new_covered[uncovered[new_covered]] if new_covered.size > 0 else np.array([], dtype=np.int32)
        if newly_uncovered.size > 0:
            uncovered[newly_uncovered] = False
            uncovered_count -= len(newly_uncovered)
            # For each newly covered t-subset, decrement gain of all blocks that cover it
            for idx in newly_uncovered:
                w = t_weights[idx]
                covering_blocks = t_to_blocks[idx]
                # Vectorized subtraction
                mask = np.array([not selected_mask[b] for b in covering_blocks])
                if mask.any():
                    gain[covering_blocks] -= w

    result = [list(k_blocks[i]) for i in selected]

    # Post-processing 1: Remove redundant blocks
    if elapsed() < deadline - 25 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 25:
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 25:
                    break
                block_idx = selected[idx_remove]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(idx_remove)
                    result = [list(k_blocks[i]) for i in selected]
                    changed = True

    # Post-processing 2: Optimized Local Search with Bitsets
    if elapsed() < deadline - 15 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        selected_set = set(selected)
        improved = True
        max_rounds = 100
        
        for _ in range(max_rounds):
            if elapsed() > deadline - 15:
                break
            if not improved:
                break
            improved = False

            for si in range(len(selected)):
                if elapsed() > deadline - 15:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                drop_set_arr = block_cover_arrays[drop_block]
                
                # Find critical t-subsets
                critical = []
                for idx in drop_set:
                    if cover_count[idx] == 1:
                        critical.append(idx)

                if not critical:
                    # Can drop without replacement
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    break

                # Need replacement: use safety-margin heuristic
                # Maximize the minimum new coverage count among critical t-subsets
                critical_bits = 0
                for c in critical:
                    critical_bits |= (1 << c)
                
                critical_arr = np.array(critical, dtype=np.int32)
                critical_set = set(critical)

                best_candidate = -1
                best_min_margin = -1
                best_new_critical_count = float('inf')

                for b in range(num_k):
                    if b in selected_set:
                        continue
                    # Check if b covers all criticals
                    if (block_bits[b] & critical_bits) != critical_bits:
                        continue
                    
                    b_set = block_t_subsets[b]
                    b_arr = block_cover_arrays[b]
                    
                    # Compute new coverage counts for critical t-subsets
                    # For each critical idx: new_count = cover_count[idx] - 1 + 1 = cover_count[idx] (since it's in both A and B)
                    # Wait, that's not right. Let me recompute:
                    # If idx in A (drop_set) and idx in B (b_set): new = C - 1 + 1 = C
                    # If idx in A and not in B: new = C - 1
                    # If idx not in A and in B: new = C + 1
                    # Critical means C == 1, so:
                    # If idx in A and B: new = 1 (still critical)
                    # If idx in A and not B: new = 0 (uncovered - invalid!)
                    # So we must have all criticals in B, which we already checked.
                    # All criticals will have new count = 1.
                    
                    # Now compute how many NEW criticals are created (count becomes 1)
                    # and what's the minimum new count overall
                    new_critical_count = 0
                    min_new_count = float('inf')
                    
                    # Check idx in B that are not in A
                    b_not_in_a = b_set - drop_set
                    for idx in b_not_in_a:
                        new_c = cover_count[idx] + 1
                        if new_c == 1:
                            new_critical_count += 1
                        if new_c < min_new_count:
                            min_new_count = new_c
                    
                    # Check idx in A that are not in B
                    a_not_in_b = drop_set - b_set
                    for idx in a_not_in_b:
                        new_c = cover_count[idx] - 1
                        if new_c == 1:
                            new_critical_count += 1
                        if new_c < min_new_count:
                            min_new_count = new_c
                    
                    # Also check criticals (they stay at count 1)
                    # min_new_count should consider them too
                    min_new_count = min(min_new_count, 1)
                    
                    # Score: maximize min_new_count, then minimize new_critical_count
                    # But min_new_count is always <= 1 for valid swaps (criticals stay at 1)
                    # So we just minimize new_critical_count
                    if new_critical_count < best_new_critical_count:
                        best_new_critical_count = new_critical_count
                        best_candidate = b

                if best_candidate != -1:
                    # Perform swap
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    break

    # Post-processing 3: Final drops
    if elapsed() < deadline - 10 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 10:
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 10:
                    break
                block_idx = selected[idx_remove]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(idx_remove)
                    result = [list(k_blocks[i]) for i in selected]
                    changed = True

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(selected)
            for i in range(num_k):
                if elapsed() > deadline - 5:
                    break
                if i in selected_set:
                    continue
                has_uncovered = False
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        has_uncovered = True
                        break
                if has_uncovered:
                    selected.append(i)
                    selected_set.add(i)
                    for idx2 in block_t_subsets[i]:
                        final_cover[idx2] += 1
            result = [list(k_blocks[i]) for i in selected]
    elif num_t > 0:
        uncovered = np.ones(num_t, dtype=np.bool_)
        for i in range(num_k):
            if elapsed() > deadline - 5:
                break
            has_uncovered = False
            for idx in block_t_subsets[i]:
                if uncovered[idx]:
                    has_uncovered = True
                    break
            if has_uncovered:
                selected.append(i)
                for idx2 in block_t_subsets[i]:
                    uncovered[idx2] = False
            if not uncovered.any():
                break
        result = [list(k_blocks[i]) for i in selected]

    return result
