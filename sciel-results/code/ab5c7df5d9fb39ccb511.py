    def try_delete_safe(i):
        L = len(current_word)
        if L < n:
            return False
            
        affected_old = []
        for j in range(max(0, i - n + 1), min(i, L - n) + 1):
            wc = window_code(current_word, j)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                affected_old.append(pidx)
                
        # Subtract
        for pidx in affected_old:
            counts[pidx] -= 1
            
        deleted_val = current_word[i]
        current_word.pop(i)
        
        new_L = L - 1
        affected_new = []
        for j in range(max(0, i - n + 1), min(i, new_L - n) + 1):
            wc = window_code(current_word, j)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                affected_new.append(pidx)
                counts[pidx] += 1
        
        # Check validity
        valid = True
        all_affected = set(affected_old + affected_new)
        for pidx in all_affected:
            if counts[pidx] == 0:
                valid = False
                break
        
        if valid:
            if new_L < best_len:
                best_len = new_L
                best_word = current_word[:]
            return True
        else:
            # Revert
            current_word.insert(i, deleted_val)
            for pidx in affected_old:
                counts[pidx] += 1
            for pidx in affected_new:
                counts[pidx] -= 1
            return False
