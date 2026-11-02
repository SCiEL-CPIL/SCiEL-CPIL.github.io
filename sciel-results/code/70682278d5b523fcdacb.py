            if best_gain == 0:
                # Fallback to full scan
                for i in range(num_blocks):
                    gain = uncovered[block_t_indices[i]].sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_gain == 0:
                    break
