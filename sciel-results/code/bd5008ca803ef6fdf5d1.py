    def gen_row(current_row, last_val, diff_mask):
        if len(current_row) == k + 1:
            rows_data.append((current_row[:], diff_mask))
            return
        
        for next_val in range(last_val + 1, M + 1):
            valid = True
            new_mask = diff_mask
            for a in current_row:
                d = next_val - a
                if (new_mask >> d) & 1:
                    valid = False
                    break
                new_mask |= (1 << d)
            
            if not valid:
                continue
                
            current_row.append(next_val)
            gen_row(current_row, next_val, new_mask)
            current_row.pop()
