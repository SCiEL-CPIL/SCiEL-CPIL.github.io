import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

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
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    if elapsed() > deadline - 30:
        return [list(b) for b in k_blocks[:1]]

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # t_cover_count: how many blocks cover each t-subset
    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)

    # Step 4: Weighted greedy set cover
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    t_weights = 1.0 / safe_counts

    # Initialize gain for each block
    gain = np.zeros(num_k, dtype=np.float64)
    for i, covered in enumerate(block_t_subsets):
        if covered:
            gain[i] = np.sum(t_weights[list(covered)])

    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)

    # Greedy loop with numpy argmax for speed
    while uncovered_count > 0 and elapsed() < deadline - 20:
        masked_gain = gain.copy()
        masked_gain[selected_mask] = -1.0
        best_idx = int(np.argmax(masked_gain))
        best_gain = gain[best_idx]

        if best_gain <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

        # Update uncovered and gains
        for idx in block_t_subsets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = t_weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w

    # Convert to actual blocks
    result = [list(k_blocks[i]) for i in selected]

    # Post-processing 1: Remove redundant blocks (iterative single-drop)
    if elapsed() < deadline - 15 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 15:
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 15:
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

    # Post-processing 2: Local search with swaps
    if elapsed() < deadline - 10 and result:
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        selected_set = set(selected)
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if elapsed() > deadline - 10:
                break
            if not improved:
                break
            improved = False

            for si in range(len(selected)):
                if elapsed() > deadline - 10:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]

                critical = []
                for idx in drop_set:
                    if cover_count[idx] == 1:
                        critical.append(idx)

                if not critical:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    break

                best_candidate = -1
                best_new_cov = -1

                for b in range(num_k):
                    if b in selected_set:
                        continue
                    b_set = block_t_subsets[b]
                    covers_all = all(c in b_set for c in critical)
                    if not covers_all:
                        continue
                    new_unique = 0
                    for idx in b_set:
                        if cover_count[idx] == 0:
                            new_unique += 1
                    if new_unique > best_new_cov:
                        best_new_cov = new_unique
                        best_candidate = b

                if best_candidate != -1:
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

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(selected)
            for i in range(num_k):
                if elapsed() > deadline - 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in selected]
    elif num_t > 0:
        uncovered = np.ones(num_t, dtype=np.bool_)
        for i in range(num_k):
            if elapsed() > deadline - 3:
                break
            for idx in block_t_subsets[i]:
                if uncovered[idx]:
                    selected.append(i)
                    for idx2 in block_t_subsets[i]:
                        uncovered[idx2] = False
                    break
            if not uncovered.any():
                break
        result = [list(k_blocks[i]) for i in selected]

    return result
