            if not improved and T > T_min:
                # Kick move
                if len(reduced) > 1:
                    # Add a random block
                    avail = [bi for bi in range(n_blocks) if bi not in reduced]
                    if avail:
                        new_b = avail[rng.randint(0, len(avail)-1)]
                        for idx in block_tsubs[new_b]:
                            cover_count[idx] += 1
                        reduced.append(new_b)
                        
                        # Try to remove redundant
                        changed = True
                        while changed:
                            changed = False
                            double_cover = cover_count_to_bitset(cover_count)
                            for i in range(len(reduced)):
                                U_i = block_bitsets[reduced[i]] & ~double_cover
                                if not np.any(U_i):
                                    b_idx = reduced[i]
                                    for idx in block_tsubs[b_idx]:
                                        cover_count[idx] -= 1
                                    reduced.pop(i)
                                    changed = True
                                    break
                                    
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        improved = True
