    def search(available_mask, current_set_mask, forbidden_mask, current_size):
        nonlocal best_size, best_mask
        if time.time() > deadline - 2:
            return
        
        remaining = available_mask & ~forbidden_mask
        if current_size + remaining.bit_count() <= best_size:
            return
        
        if remaining == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_set_mask
            return

        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        new_forbidden = forbidden_mask
        cs = current_set_mask
        while cs:
            lsb_u = cs & -cs
            u = lsb_u.bit_length() - 1
            new_forbidden |= bad_with_pair[u][v]
            cs ^= lsb_u
        
        new_set = current_set_mask | lsb
        new_avail = available_mask & ~lsb
        
        search(new_avail, new_set, new_forbidden, current_size + 1)
        
        # Branch 2: Exclude v
        new_avail_excl = available_mask & ~lsb
        search(new_avail_excl, current_set_mask, forbidden_mask, current_size)
