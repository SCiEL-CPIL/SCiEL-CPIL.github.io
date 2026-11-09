            if current_L > 0:
                prefix_mods = x_mods[i, :current_L]
                delta_prefix = (prefix_mods == new_r).astype(np.int64) - (prefix_mods == old_r).astype(np.int64)
                new_prefix_counts = counts[:current_L] + delta_prefix
                if np.min(new_prefix_counts) < 2:
                    continue
            # else: current_L is 0, no prefix to check
