    def dfs(remaining_mask, parts_left):
        if remaining_mask == 0:
            return parts_left == 0
        
        if parts_left == 0:
            return False
        
        if remaining_mask.bit_count() > parts_left * max_vs[remaining_mask]:
            return False
            
        lsb = remaining_mask & -remaining_mask
        pivot = lsb.bit_length() - 1
        
        for sub in valid_subs:
            if (sub & remaining_mask) != sub:
                continue
            if not (sub & (1 << pivot)):
                continue
            
            path.append(sub)
            if dfs(remaining_mask ^ sub, parts_left - 1):
                return True
            path.pop()
        
        return False
