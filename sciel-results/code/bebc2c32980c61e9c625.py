            cover_count[ts_arr_i] -= 1
            cover_count[new_ts_arr] += 1

            if np.min(cover_count[ts_arr_i]) >= 1:
                # Swap is valid
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                # Revert
                cover_count[new_ts_arr] -= 1
                cover_count[ts_arr_i] += 1
