    def search(uncovered_mask, used_mask, current_centers, depth):
        nonlocal best_size, best_centers
        
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_centers = current_centers.copy()
            return True
            
        rem = uncovered_mask.bit_count()
        lb = (rem + max_cover_size - 1) // max_cover_size
        if depth + lb >= best_size:
            return False
            
        min_cand = float('inf')
        chosen_word = -1
        u = uncovered_mask
        while u:
            lsb = u & -u
            w = lsb.bit_length() - 1
            avail = (cover_masks[w] & ~used_mask).bit_count()
            if avail < min_cand:
                min_cand = avail
                chosen_word = w
            if avail == 0:
                return False
            u ^= lsb
            
        cand_mask = cover_masks[chosen_word] & ~used_mask
        
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & -cm
            c = lsb.bit_length() - 1
            gain = (cover_masks[c] & uncovered_mask).bit_count()
            candidates.append((gain, c))
            cm ^= lsb
            
        candidates.sort(reverse=True)
        
        for gain, c in candidates:
            current_centers.append(c)
            search(uncovered_mask & ~cover_masks[c], used_mask | (1 << c), current_centers, depth + 1)
            current_centers.pop()
            
        return False
