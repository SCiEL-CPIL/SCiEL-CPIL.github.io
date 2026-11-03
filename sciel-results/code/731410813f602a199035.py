    def dfs_min_max(partial_row, forb, k, max_limit, time_limit, node_count=0):
        if time.time() > time_limit or node_count > 20000:
            return None
        if len(partial_row) == k + 1:
            return partial_row
        
        last = partial_row[-1]
        partial_diffs = set()
        for i in range(len(partial_row)):
            for j in range(i+1, len(partial_row)):
                partial_diffs.add(partial_row[j] - partial_row[i])
        
        x = last + 1
        while x <= max_limit:
            conflict = False
            for a in partial_row:
                diff = x - a
                if diff in forb or diff in partial_diffs:
                    conflict = True
                    break
            
            if not conflict:
                res = dfs_min_max(partial_row + [x], forb, k, max_limit, time_limit, node_count + 1)
                if res:
                    return res
            
            x += 1
            
        return None
