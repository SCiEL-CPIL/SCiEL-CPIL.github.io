    # Final polish: Fast Local Search
    current_masks = list(best_solution)
    used_masks = set(current_masks)
    sum_comp = init_sum_comp(current_masks)
    current_score = best_score
    
    while time.time() < deadline - 2.0:
        improved = False
        for idx in range(m):
            if time.time() > deadline - 2.0:
                break
            old_mask = current_masks[idx]
            best_new_mask = -1
            best_delta = 0
            old_base = old_mask * num_masks
            sc_old = sum_comp[old_mask]
            
            for mask in range(num_masks):
                if mask in used_masks:
                    continue
                delta = (sum_comp[mask] - comp[old_base + mask]) - sc_old
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = mask
            
            if best_new_mask != -1 and best_delta > 0:
                new_mask = best_new_mask
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += best_delta
                
                # Update sum_comp
                new_base = new_mask * num_masks
                for j_idx in range(m):
                    if j_idx == idx:
                        continue
                    j_mask = current_masks[j_idx]
                    sum_comp[j_mask] += comp[new_base + j_mask] - comp[old_base + j_mask]
                
                sum_comp[new_mask] -= comp[old_base + new_mask]
                sum_comp[old_mask] = 0
                
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
        
        if not improved:
            break
    