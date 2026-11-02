<<<<<<< SEARCH
    # Run local search iterations
    for _ in range(40):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Final SA polish
    if time.time() < deadline - 5.0:
        polished = sa_polish(best_solution, max_iters=2000)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)

    return best_solution
=======
    # Run local search iterations with cached state
    def build_cover_state(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                block_ts_indices.append(block_to_ts[idx])
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
                for ts_idx in ts_list:
                    cover_count[ts_idx] += 1
        return cover_count, block_ts_indices, sol_block_indices

    def optimize_redundancy_fast(blocks, cover_count, block_ts_indices, sol_block_indices):
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 3.0:
                break
            for i in range(len(blocks)):
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    old_idx = get_block_idx(blocks[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    changed = True
                    break
        return cover_count, block_ts_indices, sol_block_indices

    # Build initial state
    cover_count, block_ts_indices, sol_block_indices = build_cover_state(best_solution)
    cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
        best_solution, cover_count, block_ts_indices, sol_block_indices)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    # Local search with shared state
    for _ in range(60):
        if time.time() > deadline - 8.0:
            break

        # Single swap
        improved = False
        n = len(best_solution)
        for i in range(n):
            if time.time() > deadline - 4.0:
                break
            unique_ts = [ts_idx for ts_idx in block_ts_indices[i] if cover_count[ts_idx] == 1]
            if not unique_ts:
                continue
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            if not candidates:
                continue
            valid = [c for c in candidates if c not in sol_block_indices]
            if not valid:
                continue
            new_idx = random.choice(valid)
            old_idx = get_block_idx(best_solution[i])

            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_idx]:
                cover_count[ts_idx] += 1
            if old_idx >= 0:
                sol_block_indices.discard(old_idx)
            sol_block_indices.add(new_idx)
            best_solution[i] = list(all_k_blocks[new_idx])
            block_ts_indices[i] = block_to_ts[new_idx]
            improved = True
            break

        if not improved:
            # Two-exchange: remove one, add two
            n = len(best_solution)
            for i in range(n):
                if time.time() > deadline - 4.0:
                    break
                unique_ts_i = [ts_idx for ts_idx in block_ts_indices[i] if cover_count[ts_idx] == 1]
                if not unique_ts_i:
                    continue
                # Try to replace with a single block that covers all unique
                candidates = set(ts_to_blocks[unique_ts_i[0]])
                for ts_idx in unique_ts_i[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                valid = [c for c in candidates if c not in sol_block_indices]
                if valid:
                    new_idx = random.choice(valid)
                    old_idx = get_block_idx(best_solution[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_idx]:
                        cover_count[ts_idx] += 1
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    sol_block_indices.add(new_idx)
                    best_solution[i] = list(all_k_blocks[new_idx])
                    block_ts_indices[i] = block_to_ts[new_idx]
                    improved = True
                    break

        if improved:
            cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
                best_solution, cover_count, block_ts_indices, sol_block_indices)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try removing a redundant block
            removed = False
            for i in range(len(best_solution) - 1, -1, -1):
                if time.time() > deadline - 4.0:
                    break
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    old_idx = get_block_idx(best_solution[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    best_solution.pop(i)
                    block_ts_indices.pop(i)
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    removed = True
                    break
            if not removed:
                break

    # Final SA polish with state tracking
    if time.time() < deadline - 8.0:
        # Rebuild state from current best
        cover_count, block_ts_indices, sol_block_indices = build_cover_state(best_solution)
        cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
            best_solution, cover_count, block_ts_indices, sol_block_indices)

        best_len_sa = len(best_solution)
        best_blocks_sa = list(best_solution)
        current_len = len(best_solution)

        T = 50.0
        T_min = 0.5
        cooling = 0.998
        max_sa_iters = 3000

        for it in range(max_sa_iters):
            if time.time() > deadline - 3.0:
                break

            uncovered = np.where(cover_count == 0)[0]

            if len(uncovered) > 0 and random.random() < 0.5:
                # Add a block to cover an uncovered t-subset
                ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue
                valid = [c for c in candidate_blocks if c not in sol_block_indices]
                if not valid:
                    continue
                new_idx = valid[random.randint(0, len(valid) - 1)]
                new_covered = block_to_ts_np[new_idx]

                delta_uncovered = 0
                for ts_idx2 in new_covered:
                    if cover_count[ts_idx2] == 0:
                        delta_uncovered -= 1

                delta_obj = delta_uncovered * 1000 + 1
                if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                    for ts_idx2 in new_covered:
                        cover_count[ts_idx2] += 1
                    sol_block_indices.add(new_idx)
                    best_solution.append(list(all_k_blocks[new_idx]))
                    block_ts_indices.append(block_to_ts[new_idx])
                    current_len += 1
            else:
                # Remove or swap a low-criticality block
                if not sol_block_indices:
                    continue

                # Find block with fewest unique t-subsets
                min_crit = float('inf')
                min_idx = -1
                for bi, b_idx in enumerate(sol_block_indices):
                    crit = 0
                    for ts_idx2 in block_to_ts_np[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    if crit < min_crit:
                        min_crit = crit
                        min_idx = bi

                if min_idx < 0:
                    continue

                b_idx = sol_block_indices[min_idx]
                old_covered = block_to_ts_np[b_idx]

                if min_crit == 0:
                    # Safe removal
                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(b_idx)
                    # Find and remove from lists
                    for i in range(len(best_solution)):
                        if get_block_idx(best_solution[i]) == b_idx:
                            best_solution.pop(i)
                            block_ts_indices.pop(i)
                            break
                    current_len -= 1
                else:
                    # Try swap
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]
                    if unique_ts:
                        candidates = set(ts_to_blocks[unique_ts[0]])
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks[ts_idx2])
                            if not candidates:
                                break
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = random.choice(valid)
                            new_covered = block_to_ts_np[new_idx]

                            delta_uncovered = 0
                            for ts_idx2 in old_covered:
                                if cover_count[ts_idx2] == 1:
                                    delta_uncovered += 1
                            for ts_idx2 in new_covered:
                                if cover_count[ts_idx2] == 0:
                                    delta_uncovered -= 1

                            delta_obj = delta_uncovered * 1000
                            if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                                for ts_idx2 in old_covered:
                                    cover_count[ts_idx2] -= 1
                                for ts_idx2 in new_covered:
                                    cover_count[ts_idx2] += 1
                                sol_block_indices.discard(b_idx)
                                sol_block_indices.add(new_idx)
                                for i in range(len(best_solution)):
                                    if get_block_idx(best_solution[i]) == b_idx:
                                        best_solution[i] = list(all_k_blocks[new_idx])
                                        block_ts_indices[i] = block_to_ts[new_idx]
                                        break

            # Periodic redundancy cleanup
            if it % 50 == 0:
                cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
                    best_solution, cover_count, block_ts_indices, sol_block_indices)
                current_len = len(best_solution)

            if len(uncovered) == 0 and current_len < best_len_sa:
                best_len_sa = current_len
                best_blocks_sa = list(best_solution)

            T = max(T_min, T * cooling)

        if best_len_sa < best_len:
            best_solution = best_blocks_sa
            best_len = best_len_sa

    return best_solution
>>>>>>> REPLACE

