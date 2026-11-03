    def dfs_search(remaining_rows, k, used, current_max, rows, best_max, deadline):
        if time.time() > deadline:
            return False
        if remaining_rows == 0:
            return True
            
        row = [0]
        
        def build(pos, row_so_far, used_local, max_local):
            nonlocal deadline
            if time.time() > deadline:
                return False
            if len(row_so_far) == k + 1:
                rows.append(row_so_far)
                success = dfs_search(remaining_rows - 1, k, used_local, max_local, rows, best_max, deadline)
                rows.pop()
                if success:
                    return True
                return False
            
            start_x = row_so_far[-1] + 1
            if start_x > best_max:
                return False
            
            for x in range(start_x, best_max + 1):
                new_diffs = {x - a for a in row_so_far}
                if new_diffs & used_local:
                    continue
                
                row_so_far.append(x)
                used_new = used_local | new_diffs
                max_new = max(max_local, x)
                
                if max_new >= best_max:
                    row_so_far.pop()
                    continue
                
                if build(pos + 1, row_so_far, used_new, max_new):
                    return True
                    
                row_so_far.pop()
                
            return False
            
        return build(1, row, used, current_max)
