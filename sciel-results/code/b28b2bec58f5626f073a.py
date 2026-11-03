def find_sidon_of_size(n, target_size):
    # Backtracking to find any Sidon set of size target_size
    # Returns the set or None
    
    def backtrack(start, current_set, used_sums, remaining):
        if remaining == 0:
            return current_set.copy()
        
        # Pruning: if not enough numbers left
        if n - start < remaining:
            return None
            
        for i in range(start, n - remaining + 1):
            valid = True
            new_sums = []
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
                else:
                    new_sums.append(s_ii)
            
            if valid:
                current_set.append(i)
                used_sums.update(new_sums)
                
                res = backtrack(i + 1, current_set, used_sums, remaining - 1)
                if res is not None:
                    return res
                
                current_set.pop()
                used_sums.difference_update(new_sums)
                
        return None

    return backtrack(0, [], set(), target_size)
