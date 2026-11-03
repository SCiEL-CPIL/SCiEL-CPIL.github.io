    def limited_search(M, time_limit):
        def dfs(row_idx, point_idx, current_row, used_mask, rows_so_far):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return rows_so_far
            
            if point_idx == k:
                return dfs(row_idx + 1, 0, [0], used_mask, rows_so_far + [list(current_row)])
            
            last_val = current_row[-1]
            
            for x in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_used = used_mask
                for d in new_diffs:
                    new_used |= (1 << d)
                
                # Mutate
                current_row.append(x)
                res = dfs(row_idx, point_idx + 1, current_row, new_used, rows_so_far)
                current_row.pop() # Backtrack
                
                if res is not None:
                    return res
            
            return None
        
        return dfs(0, 0, [0], 0, [])
