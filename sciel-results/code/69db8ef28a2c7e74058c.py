def local_search(rows, initial_max, time_remaining):
    def get_diffs_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    used_mask = 0
    for row in rows:
        used_mask |= get_diffs_mask(row)
        
    current_max = initial_max
    best_rows = [row[:] for row in rows]
    best_max = current_max
    
    stall_count = 0
    deadline = time.time() + time_remaining
    
    while time.time() < deadline:
        # Find max row
        max_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
        current_row_max = max(rows[max_idx])
        
        if current_row_max < best_max:
            best_max = current_row_max
            best_rows = [row[:] for row in rows]
            stall_count = 0
        else:
            stall_count += 1
            
        if stall_count > 50:
            # Perturb
            # Pick a random row, try to expand it
            # Or just break and restart?
            # Let's try to find a valid row for max_idx with a LARGER max mark
            # to free up differences
            row_diffs_mask = get_diffs_mask(rows[max_idx])
            used_mask &= ~row_diffs_mask
            
            # Try max_mark = current_row_max + 5
            new_row = find_valid_row(used_mask, current_row_max + 5, 5)
            if new_row and max(new_row) > current_row_max:
                rows[max_idx] = new_row
                used_mask |= get_diffs_mask(new_row)
                stall_count = 0
            else:
                # Put back
                used_mask |= row_diffs_mask
                # If still stuck, force a restart of the local search logic
                # by randomizing a row's structure
                if random.random() < 0.1:
                     # Re-greedy one row?
                     pass

        # Try to shrink max_idx
        row_diffs_mask = get_diffs_mask(rows[max_idx])
        used_mask &= ~row_diffs_mask
        
        # Try to find a row with max < current_row_max
        new_row = find_valid_row(used_mask, current_row_max - 1, 5)
        
        if new_row:
            rows[max_idx] = new_row
            used_mask |= get_diffs_mask(new_row)
        else:
            # Try to find a different row with same max
            new_row = find_valid_row(used_mask, current_row_max, 5)
            if new_row:
                new_mask = get_diffs_mask(new_row)
                if new_mask != row_diffs_mask:
                    rows[max_idx] = new_row
                    used_mask |= new_mask
                else:
                    used_mask |= row_diffs_mask
            else:
                used_mask |= row_diffs_mask
                # Try to improve other rows
                if random.random() < 0.5:
                    other_idx = random.choice([i for i in range(len(rows)) if i != max_idx])
                    other_diffs_mask = get_diffs_mask(rows[other_idx])
                    used_mask &= ~other_diffs_mask
                    new_other = find_valid_row(used_mask, max(rows[other_idx]) - 1, 5)
                    if new_other:
                        rows[other_idx] = new_other
                        used_mask |= get_diffs_mask(new_other)
                    else:
                        used_mask |= other_diffs_mask

    return best_rows
