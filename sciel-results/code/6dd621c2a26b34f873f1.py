            # Generate new row using greedy, avoiding current global_diffs
            new_row = [0]
            last = 0
            limit = best_max + 10
            found = True
            for _ in range(num_marks - 1):
                next_val = last + 1
                placed = False
                while next_val <= limit:
                    new_diffs = {next_val - a for a in new_row}
                    if len(new_diffs) == len(new_row): # No internal duplicates
                        # Check external
                        if not any(d in global_diff_counts for d in new_diffs):
                            new_row.append(next_val)
                            last = next_val
                            placed = True
                            break
                    next_val += 1
                if not placed:
                    found = False
                    break
