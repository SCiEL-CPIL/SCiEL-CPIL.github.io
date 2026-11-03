<<<<<<< SEARCH
    # 5. Final polish: one more pass of drops + swaps with tighter time budget
    if elapsed() < deadline - 2:
        improved2 = True
        while improved2 and elapsed() < deadline - 2:
            improved2 = False
            if try_drop_blocks():
                improved2 = True
                continue
            # Quick swap pass
            for i in range(len(selected)):
                if elapsed() > deadline - 2:
                    break
                b_out = selected[i]
                unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
                if not unique_subs:
                    continue
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= t_to_blocks[s_idx]
                if not candidates:
                    continue
                candidates.discard(b_out)
                if not candidates:
                    continue
                b_in = next(iter(candidates))
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                selected[i] = b_in
                if try_drop_blocks():
                    improved2 = True
                    break
                else:
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # Final result
    result = [list(k_blocks[i]) for i in selected]
    return result
=======
    # 5. Final polish: one more pass of drops + swaps with tighter time budget
    if elapsed() < deadline - 2:
        improved2 = True
        while improved2 and elapsed() < deadline - 2:
            improved2 = False
            if try_drop_blocks():
                improved2 = True
                continue
            # Quick swap pass
            for i in range(len(selected)):
                if elapsed() > deadline - 2:
                    break
                b_out = selected[i]
                unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
                if not unique_subs:
                    continue
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= t_to_blocks[s_idx]
                if not candidates:
                    continue
                candidates.discard(b_out)
                if not candidates:
                    continue
                b_in = next(iter(candidates))
                if b_in in selected:
                    continue
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                selected[i] = b_in
                if try_drop_blocks():
                    improved2 = True
                    break
                else:
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # 6. Multi-start: run additional greedy+polish iterations if time permits
    best_selected = list(selected)
    best_count = len(selected)
    
    # Determine how many additional restarts we can afford
    time_per_restart = elapsed() / 1.0  # rough estimate from first run
    remaining = deadline - 5 - elapsed()
    max_restarts = max(1, int(remaining / max(time_per_restart * 0.7, 1.0)))
    
    for restart in range(max_restarts):
        if elapsed() > deadline - 10:
            break
        # Run greedy with random tie-breaking
        seed_val = rng.randint(0, 2**31 - 1)
        new_selected = run_greedy(tie_break_seed=seed_val)
        
        if len(new_selected) >= best_count:
            continue
        
        # Build cover count and polish
        new_cover = np.zeros(num_t, dtype=np.int32)
        for b_idx in new_selected:
            for s_idx in block_t_indices[b_idx]:
                new_cover[s_idx] += 1
        
        # Redundancy removal
        new_selected, new_cover, _ = remove_redundant(new_selected, new_cover)
        
        if len(new_selected) >= best_count:
            continue
        
        # Local search on new solution
        # Reuse the local search logic by temporarily swapping state
        saved_selected = selected
        saved_cover = cover_count
        selected = new_selected
        cover_count = new_cover
        
        improved3 = True
        while improved3 and elapsed() < deadline - 5:
            improved3 = False
            if try_drop_blocks():
                improved3 = True
                continue
            swap_found3 = False
            for i in range(len(selected)):
                if elapsed() > deadline - 5:
                    break
                b_out = selected[i]
                unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
                if not unique_subs:
                    continue
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= t_to_blocks[s_idx]
                if not candidates:
                    continue
                candidates.discard(b_out)
                if not candidates:
                    continue
                cand_list = list(candidates)
                if len(cand_list) > 15:
                    rng.shuffle(cand_list)
                    cand_list = cand_list[:15]
                for b_in in cand_list:
                    if elapsed() > deadline - 5:
                        break
                    if b_in in selected:
                        continue
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] += 1
                    selected[i] = b_in
                    if try_drop_blocks():
                        improved3 = True
                        swap_found3 = True
                        break
                    else:
                        for s_idx in block_t_indices[b_in]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_t_indices[b_out]:
                            cover_count[s_idx] += 1
                        selected[i] = b_out
                if swap_found3:
                    break
        
        # Restore and compare
        selected = saved_selected
        cover_count = saved_cover
        
        if len(new_selected) < best_count:
            best_selected = new_selected
            best_count = len(best_selected)
    
    # Final result
    result = [list(k_blocks[i]) for i in best_selected]
    return result
>>>>>>> REPLACE
