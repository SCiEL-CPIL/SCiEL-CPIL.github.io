    def local_search(mask):
        improved = True
        while improved:
            improved = False
            
            # 1. Add moves
            added = True
            while added:
                added = False
                for i in range(N):
                    if not ((mask >> i) & 1):
                        if (conflict[i] & mask) == 0:
                            mask |= (1 << i)
                            added = True
                            improved = True
                            break # Restart add loop

            # 2. Swap moves (1-swap)
            swapped = True
            while swapped:
                swapped = False
                for u in range(N):
                    if not ((mask >> u) & 1):
                        continue
                    mask_wo_u = mask & ~(1 << u)
                    
                    best_v = -1
                    best_w = weights[u]
                    
                    for v in range(N):
                        if (mask >> v) & 1:
                            continue
                        if (conflict[v] & mask_wo_u) == 0:
                            if weights[v] > best_w:
                                best_w = weights[v]
                                best_v = v
                    
                    if best_v != -1:
                        mask = (mask & ~(1 << u)) | (1 << best_v)
                        swapped = True
                        improved = True
                        break # Restart swap loop

            # 3. 2-Swap moves
            if not improved and time.time() < deadline - 5.0:
                # Try a limited number of 2-swaps
                # To avoid O(N^4) explosion, we can be selective
                # Or just try all u1, u2 but efficiently find best v1, v2
                found_2swap = False
                indices_in_mask = [i for i in range(N) if (mask >> i) & 1]
                
                for i1 in range(len(indices_in_mask)):
                    for i2 in range(i1 + 1, len(indices_in_mask)):
                        u1 = indices_in_mask[i1]
                        u2 = indices_in_mask[i2]
                        
                        mask_wo_2 = mask & ~(1 << u1) & ~(1 << u2)
                        target_weight = weights[u1] + weights[u2]
                        
                        # Find candidates
                        candidates = []
                        for v in range(N):
                            if (mask >> v) & 1:
                                continue
                            if (conflict[v] & mask_wo_2) == 0:
                                candidates.append(v)
                        
                        # Find best pair in candidates
                        best_pair_w = target_weight
                        best_v1, best_v2 = -1, -1
                        
                        # If candidates is small, check all pairs
                        if len(candidates) < 50: # Heuristic threshold
                            for k1 in range(len(candidates)):
                                v1 = candidates[k1]
                                for k2 in range(k1 + 1, len(candidates)):
                                    v2 = candidates[k2]
                                    if (conflict[v1] & (1 << v2)) == 0:
                                        w_sum = weights[v1] + weights[v2]
                                        if w_sum > best_pair_w:
                                            best_pair_w = w_sum
                                            best_v1, best_v2 = v1, v2
                        else:
                            # Approximate: pick top 2 by weight, check if compatible
                            # Sort candidates by weight desc
                            candidates.sort(key=lambda x: weights[x], reverse=True)
                            # Check top few
                            for k1 in range(min(10, len(candidates))):
                                v1 = candidates[k1]
                                for k2 in range(k1 + 1, min(10, len(candidates))):
                                    v2 = candidates[k2]
                                    if (conflict[v1] & (1 << v2)) == 0:
                                        w_sum = weights[v1] + weights[v2]
                                        if w_sum > best_pair_w:
                                            best_pair_w = w_sum
                                            best_v1, best_v2 = v1, v2

                        if best_v1 != -1 and best_v2 != -1:
                            mask = (mask & ~(1 << u1) & ~(1 << u2)) | (1 << best_v1) | (1 << best_v2)
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    # Restart main loop to allow adds/1-swaps again
                    continue

        return mask
