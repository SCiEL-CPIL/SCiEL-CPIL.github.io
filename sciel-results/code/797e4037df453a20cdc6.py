import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Greedy baseline
    used_mask = 0
    rows_greedy = []
    for _ in range(n_rows):
        row, x = [0], 0
        row_mask = 0
        while len(row) < k + 1:
            x += 1
            # Calculate new differences
            new_diffs_mask = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << d
                if new_diffs_mask & bit:
                    # Duplicate difference within row? No, x is unique, a is unique.
                    # But d could be same if x-a1 == x-a2 => a1==a2, which is false.
                    pass
                new_diffs_mask |= bit
                if used_mask & bit:
                    valid = False
                    break
            if valid and (new_diffs_mask & row_mask) == 0:
                row.append(x)
                used_mask |= new_diffs_mask
                row_mask |= new_diffs_mask
        rows_greedy.append(row)
    
    best_m = max(max(row) for row in rows_greedy)
    best_rows = rows_greedy
    
    # 2. Search for better solution
    lb = n_rows * (k * (k + 1) // 2)
    
    for m_target in range(best_m - 1, lb - 1, -1):
        if time.time() > deadline:
            break
            
        def dfs_rows(row_idx, used_mask):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return []
            
            def find_row(current_row, row_mask):
                if len(current_row) == k + 1:
                    return current_row[:]
                
                last = current_row[-1]
                rem = (k + 1) - len(current_row)
                if last + rem > m_target:
                    return None
                
                for x in range(last + 1, m_target + 1):
                    new_diffs_mask = 0
                    valid = True
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        if new_diffs_mask & bit:
                            # This shouldn't happen as x is fixed and a are distinct
                            pass
                        new_diffs_mask |= bit
                        if used_mask & bit:
                            valid = False
                            break
                        if row_mask & bit:
                            valid = False
                            break
                    if valid:
                        res = find_row(current_row + [x], row_mask | new_diffs_mask)
                        if res is not None:
                            return res
                return None
            
            row = find_row([0], 0)
            if row is None:
                return None
            
            # Update used_mask
            new_used_mask = used_mask
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    new_used_mask |= (1 << d)
            
            rest = dfs_rows(row_idx + 1, new_used_mask)
            if rest is None:
                return None
            
            return [row] + rest
        
        result = dfs_rows(0, 0)
        if result is not None:
            best_m = m_target
            best_rows = result
        else:
            if time.time() > deadline:
                break
            # If search was complete and failed, no smaller M will work.
            break

    return best_rows
