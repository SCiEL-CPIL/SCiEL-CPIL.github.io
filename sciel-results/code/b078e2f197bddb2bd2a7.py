            for d in added_diffs:
                if d in used:
                    if d not in removed_diffs:
                        return False
            
            used -= removed_diffs
            used |= added_diffs
            row[j_idx] = new_val
            return True
