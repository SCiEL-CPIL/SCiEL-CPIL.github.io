                else:
                    # 3. Try 2-out, 1-in
                    best_2out_gain = 0
                    best_2out_j = -1
                    best_2out_remove_mask = 0
                    
                    for j in range(n):
                        if not_in_set_mask & (1 << j):
                            C_j = adj_local[j] & current_mask
