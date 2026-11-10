    def bb(available, current_w, current_mask, remaining_w):
        nonlocal best_weight, best_mask
        
        if time.time() - start_time > deadline:
            return
            
        if available == 0:
            if current_w > best_weight:
                best_weight = current_w
                best_mask = current_mask
            return

        # Reduction: Add isolated nodes
        changed = True
        while changed:
            changed = False
            m = available
            i = 0
            while m:
                if m & 1:
                    if (conflict_mask[i] & available) == 0:
                        # Isolated
                        current_w += weight_float[i]
                        remaining_w -= weight_float[i]
                        current_mask |= (1 << i)
                        available &= ~(1 << i)
                        changed = True
                        m = available
                        i = 0
                        break
                i += 1
                m >>= 1
        
        if available == 0:
            if current_w > best_weight:
                best_weight = current_w
                best_mask = current_mask
            return

        # Upper Bound
        ub = current_w + remaining_w
        if ub <= best_weight + 1e-15:
            return

        # Pick branch node: Highest weight
        v = -1
        max_w = -1.0
        m = available
        i = 0
        while m:
            if m & 1:
                if weight_float[i] > max_w:
                    max_w = weight_float[i]
                    v = i
            i += 1
            m >>= 1
            
        if v == -1:
            return

        # Branch 1: Include v
        # Neighbors of v in available: conflict_mask[v] & available
        neighbors = conflict_mask[v] & available
        # Weight of neighbors
        # We need to subtract weight of v and its neighbors from remaining_w
        # sum_weights(neighbors) is O(N). 
        # Can we precompute? No, 'available' changes.
        # But we can iterate bits of 'neighbors'.
        neighbor_sum = 0.0
        m = neighbors
        i = 0
        while m:
            if m & 1:
                neighbor_sum += weight_float[i]
            i += 1
            m >>= 1
            
        new_available = available & ~(1 << v) & ~neighbors
        new_remaining_w = remaining_w - weight_float[v] - neighbor_sum
        bb(new_available, current_w + weight_float[v], current_mask | (1 << v), new_remaining_w)

        # Branch 2: Exclude v
        new_available = available & ~(1 << v)
        new_remaining_w = remaining_w - weight_float[v]
        bb(new_available, current_w, current_mask, new_remaining_w)
