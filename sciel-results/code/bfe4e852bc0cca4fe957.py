                    required_point_mask = 0
                    for p in required_points:
                        required_point_mask |= (1 << p)
                    
                    found_k = -1
                    for k_idx in range(n_blocks):
                        if (block_point_masks[k_idx] & required_point_mask) == required_point_mask:
                            if (block_masks[k_idx] & required_mask) == required_mask:
                                found_k = k_idx
                                break
