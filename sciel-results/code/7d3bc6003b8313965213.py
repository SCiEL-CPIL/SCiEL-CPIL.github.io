                    new_mask = 0
                    for y in current_set:
                        d = x - y
                        new_mask |= (1 << d)
                    gain = bin((new_mask & target_mask) & ~covered_mask).count('1')
