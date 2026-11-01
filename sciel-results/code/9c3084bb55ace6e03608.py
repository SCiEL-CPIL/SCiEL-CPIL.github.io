import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin from 120s

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    # Build incidence: which t-subsets each block covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Phase 1: Greedy set cover
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    solution_blocks = []

    while True:
        if time.time() > deadline - 15.0:
            break
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1

    best_solution = solution_blocks
    best_len = len(solution_blocks)

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Redundancy removal
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
        return blocks

    # Spectral/sparsification-based local search
    # Idea: build a co-occurrence graph where edge weight between blocks i,j = number of shared t-subsets
    # Remove blocks with low "conductance" (few unique t-subsets relative to total)
    def spectral_prune(blocks, max_rounds=50):
        """
        Iteratively identify and remove structurally redundant blocks.
        A block is 'redundant' if all its t-subsets are covered by other blocks.
        We prioritize removing blocks that share the most t-subsets with remaining blocks
        (high connectivity to others = more redundant).
        """
        if len(blocks) < 2:
            return blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        # Map block list indices to global block indices
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            block_indices.append(idx)

        changed = False
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            # Find removable blocks (all t-subsets have cover_count >= 2)
            removable = []
            for i in range(len(blocks)):
                removable_i = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] <= 1:
                        removable_i = False
                        break
                if removable_i:
                    removable.append(i)

            if not removable:
                break

            # Among removable, pick the one with highest "redundancy score"
            # Score = number of t-subsets shared with other blocks (more shared = more redundant)
            best_i = None
            best_score = -1
            for i in removable:
                score = 0
                for ts in itertools.combinations(blocks[i], t):
                    score += cover_count[ts_index_map[ts]]  # higher = more shared
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is None:
                break

            # Remove block at best_i
            block_to_remove = blocks[best_i]
            for ts in itertools.combinations(block_to_remove, t):
                cover_count[ts_index_map[ts]] -= 1
            blocks.pop(best_i)
            changed = True

        return blocks

    # 1-swap: replace one block with another that covers its unique t-subsets
    def local_search_swap(blocks, max_rounds=100):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            found = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved
                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                if not unique_ts:
                    continue
                # Find a replacement block that covers all unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                if not candidates:
                    continue
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                # Pick the replacement with highest overlap (most shared t-subsets) to maximize future redundancy
                best_cand = valid_candidates[0]
                best_overlap = -1
                for c in valid_candidates:
                    overlap = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] > 0:
                            overlap += 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cand = c

                new_block_idx = best_cand
                new_block = list(all_k_blocks[new_block_idx])
                old_block = blocks[i]
                old_block_idx = get_block_idx(old_block)
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                improved = True
                found = True
                break
            if not found:
                break
        return blocks, improved

    # 2-exchange: replace two blocks with one (or two with two better ones)
    def local_search_2exchange(blocks, max_rounds=30):
        if len(blocks) < 2:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        n = len(blocks)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            found = False
            for i in range(n):
                if time.time() > deadline - 3.0:
                    return blocks, improved
                for j in range(i + 1, n):
                    if time.time() > deadline - 3.0:
                        return blocks, improved
                    # Try merging: find one block covering all unique t-subsets of blocks[i] and blocks[j]
                    unique_ts_i = []
                    for ts in itertools.combinations(blocks[i], t):
                        idx = ts_index_map[ts]
                        if cover_count[idx] == 1:
                            unique_ts_i.append(idx)
                    unique_ts_j = []
                    for ts in itertools.combinations(blocks[j], t):
                        idx = ts_index_map[ts]
                        if cover_count[idx] == 1:
                            unique_ts_j.append(idx)

                    all_unique = unique_ts_i + unique_ts_j
                    if not all_unique:
                        continue

                    # Try single block replacement
                    candidates = set(ts_to_blocks[all_unique[0]])
                    for ts_idx in all_unique[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    if candidates:
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_block_idx = random.choice(valid)
                            new_block = list(all_k_blocks[new_block_idx])
                            old_i = blocks[i]
                            old_j = blocks[j]
                            for ts in itertools.combinations(old_i, t):
                                cover_count[ts_index_map[ts]] -= 1
                            for ts in itertools.combinations(old_j, t):
                                cover_count[ts_index_map[ts]] -= 1
                            for ts in itertools.combinations(new_block, t):
                                cover_count[ts_index_map[ts]] += 1
                            sol_block_indices.discard(get_block_idx(old_i))
                            sol_block_indices.discard(get_block_idx(old_j))
                            sol_block_indices.add(new_block_idx)
                            blocks.pop(j)
                            blocks[i] = new_block
                            improved = True
                            found = True
                            break

                    # Try two-block replacement with better overlap
                    if unique_ts_i:
                        cand_i = set(ts_to_blocks[unique_ts_i[0]])
                        for ts_idx in unique_ts_i[1:]:
                            cand_i.intersection_update(ts_to_blocks[ts_idx])
                            if not cand_i:
                                break
                    else:
                        cand_i = set(range(num_k_blocks))
                    if unique_ts_j:
                        cand_j = set(ts_to_blocks[unique_ts_j[0]])
                        for ts_idx in unique_ts_j[1:]:
                            cand_j.intersection_update(ts_to_blocks[ts_idx])
                            if not cand_j:
                                break
                    else:
                        cand_j = set(range(num_k_blocks))

                    valid_i = [c for c in cand_i if c not in sol_block_indices]
                    valid_j = [c for c in cand_j if c not in sol_block_indices and c not in cand_i]
                    if valid_i and valid_j:
                        # Pick replacements with maximum overlap
                        best_pair_score = -1
                        best_pair = None
                        sample_i = random.sample(valid_i, min(5, len(valid_i)))
                        sample_j = random.sample(valid_j, min(5, len(valid_j)))
                        for ci in sample_i:
                            for cj in sample_j:
                                overlap = 0
                                for ts_idx in block_to_ts[ci]:
                                    if cover_count[ts_idx] > 0:
                                        overlap += 1
                                for ts_idx in block_to_ts[cj]:
                                    if cover_count[ts_idx] > 0:
                                        overlap += 1
                                if overlap > best_pair_score:
                                    best_pair_score = overlap
                                    best_pair = (ci, cj)
                        if best_pair:
                            new_i_idx, new_j_idx = best_pair
                            new_i = list(all_k_blocks[new_i_idx])
                            new_j = list(all_k_blocks[new_j_idx])
                            old_i = blocks[i]
                            old_j = blocks[j]
                            for ts in itertools.combinations(old_i, t):
                                cover_count[ts_index_map[ts]] -= 1
                            for ts in itertools.combinations(old_j, t):
                                cover_count[ts_index_map[ts]] -= 1
                            for ts in itertools.combinations(new_i, t):
                                cover_count[ts_index_map[ts]] += 1
                            for ts in itertools.combinations(new_j, t):
                                cover_count[ts_index_map[ts]] += 1
                            sol_block_indices.discard(get_block_idx(old_i))
                            sol_block_indices.discard(get_block_idx(old_j))
                            sol_block_indices.add(new_i_idx)
                            sol_block_indices.add(new_j_idx)
                            blocks[i] = new_i
                            blocks[j] = new_j
                            improved = True
                            found = True
                            break
                if found:
                    break
            if not found:
                break
        return blocks, improved

    # WalkSAT-style: random flips with acceptance
    def walksat_local_search(blocks, max_iters=5000):
        selected = np.zeros(num_k_blocks, dtype=bool)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                selected[idx] = True
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        num_uncovered = int(np.sum(cover_count == 0))
        num_selected = int(np.sum(selected))
        penalty = 1000

        def compute_score():
            return num_selected + penalty * num_uncovered

        best_score = compute_score()
        best_selected = selected.copy()

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            if random.random() < 0.6:
                # Try turning off a selected block
                selected_indices = np.where(selected)[0]
                if len(selected_indices) == 0:
                    continue
                b_idx = selected_indices[random.randint(0, len(selected_indices) - 1)]
                would_uncover = 0
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        would_uncover += 1
                if would_uncover == 0:
                    selected[b_idx] = False
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    num_selected -= 1
                    if compute_score() < best_score:
                        best_score = compute_score()
                        best_selected = selected.copy()
                elif random.random() < 0.05:
                    selected[b_idx] = False
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    num_selected -= 1
                    num_uncovered = int(np.sum(cover_count == 0))
                    if compute_score() < best_score:
                        best_score = compute_score()
                        best_selected = selected.copy()
                    else:
                        selected[b_idx] = True
                        for ts_idx in block_to_ts[b_idx]:
                            cover_count[ts_idx] += 1
                        num_selected += 1
                        num_uncovered = int(np.sum(cover_count == 0))
            else:
                # Try turning on an unselected block
                unselected_indices = np.where(~selected)[0]
                if len(unselected_indices) == 0:
                    continue
                sample_size = min(15, len(unselected_indices))
                candidates = unselected_indices[random.sample(range(len(unselected_indices)), sample_size)]
                best_gain = -1
                best_b_idx = -1
                for b_idx in candidates:
                    gain = 0
                    for ts_idx in block_to_ts[b_idx]:
                        if cover_count[ts_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b_idx = b_idx
                if best_b_idx == -1 or best_gain == 0:
                    continue
                selected[best_b_idx] = True
                for ts_idx in block_to_ts[best_b_idx]:
                    cover_count[ts_idx] += 1
                num_selected += 1
                num_uncovered = int(np.sum(cover_count == 0))
                if compute_score() < best_score:
                    best_score = compute_score()
                    best_selected = selected.copy()

        best_blocks = [list(all_k_blocks[i]) for i in range(num_k_blocks) if best_selected[i]]
        # Ensure validity
        final_cover = np.zeros(num_t_subsets, dtype=np.int32)
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                final_cover[ts_index_map[ts]] += 1
        uncovered = np.where(final_cover == 0)[0]
        if len(uncovered) > 0:
            for ts_idx in uncovered:
                if time.time() > deadline - 2.0:
                    break
                for b_idx in ts_to_blocks[ts_idx]:
                    if not best_selected[b_idx]:
                        best_selected[b_idx] = True
                        best_blocks.append(list(all_k_blocks[b_idx]))
                        for ts2 in block_to_ts[b_idx]:
                            final_cover[ts2] += 1
                        break
        return best_blocks

    # Main optimization loop
    for iteration in range(30):
        if time.time() > deadline - 5.0:
            break

        # Spectral prune
        best_solution = spectral_prune(best_solution)

        # 1-swap
        best_solution, improved = local_search_swap(best_solution, max_rounds=50)
        if improved:
            best_solution = spectral_prune(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # 2-exchange
            best_solution, improved2 = local_search_2exchange(best_solution, max_rounds=20)
            if improved2:
                best_solution = spectral_prune(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # WalkSAT
                ws_solution = walksat_local_search(best_solution, max_iters=3000)
                ws_solution = spectral_prune(ws_solution)
                if len(ws_solution) < best_len:
                    best_len = len(ws_solution)
                    best_solution = ws_solution
                else:
                    break

    # Final redundancy removal
    best_solution = optimize_redundancy(best_solution)

    return best_solution
