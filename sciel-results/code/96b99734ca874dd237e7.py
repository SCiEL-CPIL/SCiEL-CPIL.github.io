            # Handle 2*i separately
            s_2i = 2 * i
            if target_min <= s_2i <= target_max:
                idx = s_2i - target_min
                old_c = new_counts[idx]
                new_c = old_c + 1
                new_hist[old_c] -= 1
                new_hist[new_c] += 1
                new_counts[idx] = new_c
