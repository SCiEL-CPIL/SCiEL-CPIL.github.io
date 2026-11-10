                delta = 0
                for j in triple_to_k4s[i]:
                    old_count = k4_counts[j]
                    new_count = old_count - 1 if old_c == 0 else old_count + 1
                    if old_count == 4: delta -= 1
                    if new_count == 4: delta += 1
                for j in triple_to_k5s[i]:
                    old_count = k5_counts[j]
                    new_count = old_count - 1 if old_c == 1 else old_count + 1
                    if old_count == 10: delta -= 1
                    if new_count == 10: delta += 1
