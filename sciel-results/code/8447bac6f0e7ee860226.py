    # Initial Greedy MIS
    greedy_iset = []
    remaining = valid_mask
    while remaining:
        # Pick vertex with lowest degree in remaining
        best_v = -1
        min_deg = N
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = bin(adj_mask[idx] & remaining).count('1')
            if deg < min_deg:
                min_deg = deg
                best_v = idx
            temp &= ~lsb
            
        if best_v == -1: break
        
        greedy_iset.append(best_v)
        remaining &= ~adj_mask[best_v]
        remaining &= ~(1 << best_v)
        
    best_iset_size = len(greedy_iset)
    best_iset = greedy_iset[:]
