    def full_scan_polish(masks, used_set, score):
        improved = False
        m_len = len(masks)
        for i in range(m_len):
            old_mask = masks[i]
            best_new_mask = old_mask
            best_delta = 0
            
            for new_mask in range(total_masks):
                if new_mask == old_mask or new_mask in used_set:
                    continue
                
                delta = 0
                # Local vars for speed
                om = old_mask
                nm = new_mask
                for j in range(m_len):
                    if j == i:
                        continue
                    other = masks[j]
                    # Check old
                    if (om & other) == om or (other & om) == om:
                        delta -= 1
                    # Check new
                    if (nm & other) == nm or (other & nm) == nm:
                        delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_delta > 0:
                used_set.remove(old_mask)
                masks[i] = best_new_mask
                used_set.add(best_new_mask)
                score += best_delta
                improved = True
                
        return masks, used_set, score, improved
