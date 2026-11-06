            success = False
            # Try up to 5 random pairs
            for _ in range(5):
                if len(candidates) < 2:
                    break
                y1, y2 = random.sample(candidates, 2)
                
                # Check y1
                if _can_add_np(counts, chosen, y1):
                    _add_to_counts(counts, chosen, y1)
                    chosen.append(y1)
                    chosen_set.add(y1)
                    
                    # Check y2
                    if _can_add_np(counts, chosen, y2):
                        _add_to_counts(counts, chosen, y2)
                        chosen.append(y2)
                        chosen_set.add(y2)
                        
                        # Success
                        candidates.remove(y1)
                        candidates.remove(y2)
                        success = True
                        break
                    else:
                        # Revert y1
                        _remove_from_counts(counts, chosen, y1)
                        chosen.pop()
                        chosen_set.remove(y1)
                # If y1 fails, try next pair
            
            if not success:
                # Revert x
                _add_to_counts(counts, chosen, x)
                chosen.insert(i, x)
                chosen_set.add(x)
                candidates.remove(x)
    