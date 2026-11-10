            if to_remove:
                to_remove_set = set(to_remove)
                for b_idx in to_remove:
                    for s_idx in block_subs[b_idx]:
                        cover_count[s_idx] -= 1
                selected_blocks = [b for b in selected_blocks if b not in to_remove_set]
                changed = True
