    def tighten_solution(rows, n_rows, k, used_bits, deadline):
        """
        Try to reduce the values in the rows to minimize the max entry.
        Returns the new rows and the new used_bits.
        """
        # We need to recompute used_bits for the current rows
        # But we pass it in.
        
        # To tighten, we iterate rows.
        # For each row, we iterate positions from k down to 1.
        # We try to reduce row[i] to the smallest possible value > row[i-1].
        # This value must not create any conflict with used_bits (excluding diffs from this row itself).
        
        # This is tricky because reducing a value changes the differences within the row.
        # So we must check:
        # 1. New differences within the row are distinct.
        # 2. New differences do not conflict with differences from OTHER rows.
        
        # Let's extract "other_rows_diffs" as a set or bitset.
        
        # For efficiency, we can just try values and check.
        
        # Since k is small (8), and rows are few (4), this is fast.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for i in range(k, 0, -1):
                    # Try to reduce row[i]
                    # Current value is row[i]
                    # Min possible is row[i-1] + 1
                    # We want the smallest x > row[i-1] such that:
                    # - All diffs x - row[j] (j < i) are distinct among themselves
                    # - None of these diffs are in used_bits (from other rows)
                    # - The diffs row[i]-row[j] (j < i) are removed from used_bits
                    # - The new diffs are added to used_bits
                    
                    # Note: "used_bits" passed in should represent ALL diffs.
                    # When we change row[i], we remove old diffs and add new ones.
                    
                    # Let's compute diffs for current row[i]
                    old_diffs = set()
                    for j in range(i):
                        old_diffs.add(row[i] - row[j])
                    
                    # Remove old_diffs from a temporary bitset
                    # But we need to check against "other" diffs.
                    
                    # Let's maintain a global used_bits set (or int).
                    # We will simulate the change.
                    
                    # For candidate x in range(row[i-1]+1, row[i]):
                    #   new_diffs = {x - row[j] for j in range(i)}
                    #   if len(new_diffs) != i: continue (duplicates within row)
                    #   # Check conflict with other rows
                    #   # We need a bitset of diffs from other rows.
                    #   # This is expensive to compute every time.
                    #   # Alternative: Just check against the global used_bits, 
                    #   # but we must mask out the old_diffs of this row.
                    
                    # Let's pass a `global_used` int.
                    # `other_used` = global_used ^ (bits for old_diffs of this row) 
                    # Wait, bits for old_diffs might overlap with other rows? No, by validity, they are distinct.
                    # So `other_used` = global_used & ~(bits for all diffs in this row).
                    
                    # Actually, simpler:
                    # Just iterate x downwards from row[i]-1 to row[i-1]+1.
                    # Check validity.
                    # If valid, update row[i] = x, update global_used, changed=True, break.
                    
                    # To make it fast, we pre-compute the bitset of diffs for the current row.
                    row_diff_bits = 0
                    for j in range(len(row)):
                        for l in range(j+1, len(row)):
                            row_diff_bits |= (1 << (row[l] - row[j]))
                    
                    other_used = used_bits & ~row_diff_bits
                    
                    # Now try to reduce row[i]
                    # We need to check if new x creates valid diffs.
                    # We also need to ensure new diffs don't conflict with `other_used`.
                    # And new diffs are distinct.
                    
                    # We also need to ensure that the diffs involving row[i] and row[j] (j < i) 
                    # are not in `other_used`.
                    # And they are distinct.
                    
                    # What about diffs involving row[i] and row[j] (j > i)?
                    # Those are `row[j] - x`.
                    # They must also be distinct and not in `other_used`.
                    # And they must be distinct from the "new" diffs `x - row[j] (j < i)`.
                    
                    # This is getting complex.
                    # Since k=8, we can just brute force check the whole row validity for a candidate x.
                    
                    for x in range(row[i] - 1, row[i-1], -1):
                        # Create new row
                        new_row = row[:]
                        new_row[i] = x
                        
                        # Check if new_row is strictly increasing
                        if not all(new_row[j+1] > new_row[j] for j in range(k)):
                            continue
                            
                        # Check if new_row has distinct internal diffs
                        new_row_diffs = set()
                        valid_internal = True
                        for j in range(k+1):
                            for l in range(j+1, k+1):
                                d = new_row[l] - new_row[j]
                                if d in new_row_diffs:
                                    valid_internal = False
                                    break
                                new_row_diffs.add(d)
                        if not valid_internal:
                            continue
                        
                        # Check if new_row_diffs conflict with other_used
                        # Convert other_used to set? No, check bit by bit or use bitset intersection.
                        # Let's convert new_row_diffs to a bitset
                        new_row_diff_bits = 0
                        for d in new_row_diffs:
                            new_row_diff_bits |= (1 << d)
                        
                        if new_row_diff_bits & other_used:
                            continue
                        
                        # If valid, accept
                        row[i] = x
                        # Update used_bits
                        used_bits = (used_bits & ~row_diff_bits) | new_row_diff_bits
                        changed = True
                        break # Try next position or row
                    
        return rows, used_bits
    