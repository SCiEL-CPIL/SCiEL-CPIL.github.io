    def bb(avail_mask, curr_score, curr_set):
        nonlocal best_score, best_set
        
        if not time_ok():
            return

        # Update incumbent with greedy solution from current state
        # This is a valid independent set extending curr_set
        g_score, g_set = greedy_from_avail(avail_mask)
        total_g_score = curr_score + g_score
        if total_g_score > best_score:
            best_score = total_g_score
            best_set = curr_set + g_set

        # Bound check
        # Upper bound = curr_score + sum(weights of available vertices)
        avail_sum = 0.0
        m = avail_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            avail_sum += sw[idx]
            m ^= lsb
            if curr_score + avail_sum <= best_score:
                return # Prune

        if curr_score + avail_sum <= best_score:
            return

        if avail_mask == 0:
            if curr_score > best_score:
                best_score = curr_score
                best_set = curr_set[:]
            return

        # Pick branching vertex: highest weight available (lowest index)
        lsb = avail_mask & -avail_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        new_avail_1 = avail_mask & ~(1 << v) & ~s_adj[v]
        curr_set.append(sorted_indices[v] + 1)
        bb(new_avail_1, curr_score + sw[v], curr_set)
        curr_set.pop()
        
        if not time_ok():
            return

        # Branch 2: Exclude v
        new_avail_2 = avail_mask & ~(1 << v)
        bb(new_avail_2, curr_score, curr_set)
