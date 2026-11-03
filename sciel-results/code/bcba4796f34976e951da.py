    def local_search(selected, time_limit):
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)
        
        # Precompute reverse cover for efficiency? 
        # We have global reverse_cover list of lists.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                
                # Find unique points
                unique_points = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_points.append(pos)
                    mm ^= lowbit
                
                if not unique_points:
                    # Remove w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue
                
                # Find candidates covering all unique points
                # Candidate w' must be in intersection of reverse_cover[x] for x in unique_points
                # Start with the first unique point's coverers
                candidates = set(reverse_cover[unique_points[0]])
                for x in unique_points[1:]:
                    candidates &= set(reverse_cover[x])
                    if not candidates:
                        break
                
                # Remove current selected words from candidates? 
                # We are swapping w for w'. w' should not be in selected_set (unless w' == w, which is no-op).
                # Actually, if w' is in selected_set, it's not a swap, it's a no-op or invalid.
                candidates -= selected_set
                
                if not candidates:
                    continue
                
                # Pick a candidate
                # Heuristic: Pick one that covers the most points with count == 2? 
                # Or just pick one randomly from candidates?
                # Let's pick the one that maximizes (masks[w'] & ~m).bit_count() ? 
                # Or simply pick the first one to test if it enables removals.
                # To be safe, let's try a few candidates.
                
                best_candidate = -1
                # Score candidate by how many "count==2" points it covers? 
                # If it covers points with count==2, those become count==3.
                # If we remove w, points in w\w' go from count>=2 to count-1.
                # We want to create points with count==1? No, we want to make OTHER words redundant.
                # A word v is redundant if all its points have count >= 2 (after removal of w).
                
                # Let's just pick a candidate that maximizes overlap with "weakly covered" points?
                # Simpler: Just pick one. If it doesn't lead to improvement, the loop will try other words.
                # But we need to ensure we don't get stuck.
                
                # Let's try to pick a candidate that minimizes the "loss" of coverage?
                # Actually, any valid swap maintains the cover.
                # The gain comes from the subsequent polish.
                
                # Let's pick a candidate that covers the most points that are currently covered by exactly 2 words?
                # No, let's pick a candidate that is "diverse".
                
                # Let's just pick the first candidate from the set.
                # To add variety, we can shuffle the candidates.
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                # Try the first few candidates
                for w_prime in cand_list[:5]:
                    m_prime = masks[w_prime]
                    
                    # Check if swap is valid (it is by construction of candidates)
                    # Perform swap
                    # Decrement counts for w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                        
                    # Increment counts for w_prime
                    mm = m_prime
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                        
                    selected[idx] = w_prime
                    selected_set.remove(w)
                    selected_set.add(w_prime)
                    improved = True
                    break # Try next word in main loop
                    
                if improved:
                    # Immediate polish to see if we can remove more
                    # We can call a lightweight polish here
                    # But calling full polish inside the loop is expensive.
                    # Let's just mark improved and let the outer while loop continue.
                    # The outer loop will re-shuffle and try to remove redundancies.
                    pass

        # Final polish
        selected = polish(selected, time_limit)
        return selected
    