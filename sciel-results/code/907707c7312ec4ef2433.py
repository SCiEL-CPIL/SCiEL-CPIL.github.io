            max_gain = -1
            count_max = 0
            chosen_idx = -1
            for i in range(num_k_subs):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    count_max = 1
                    chosen_idx = i
                elif gain == max_gain:
                    count_max += 1
                    # With probability 1/count_max, replace chosen_idx with i
                    # This ensures uniform distribution among all max-gain blocks seen so far
                    if random.random() < 1.0 / count_max:
                        chosen_idx = i
