    def local_search(current_set, current_score, current_n):
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            current_set_copy = set(current_set)
            sorted_in = sorted(current_set_copy)
            
            # Generate a list of potential swaps
            potential_swaps = []
            for p_in in sorted_in:
                max_p_out = (MAX_N * p_in) // current_n
                if max_p_out > 1000: max_p_out = 1000
                
                # Find valid p_out
                for p_out in PRIMES:
                    if p_out in current_set_copy:
                        continue
                    if p_out == p_in:
                        continue
                    if p_out > max_p_out:
                        break
                    potential_swaps.append((p_in, p_out))
            
            if not potential_swaps:
                break
                
            # Randomly sample swaps to try
            # If the list is small, try all. If large, sample.
            num_to_try = min(len(potential_swaps), 100)
            random.shuffle(potential_swaps)
            
            for p_in, p_out in potential_swaps[:num_to_try]:
                new_n = (current_n // p_in) * p_out
                if new_n > MAX_N or new_n < 2:
                    continue
                
                new_set = current_set_copy.copy()
                new_set.remove(p_in)
                new_set.add(p_out)
                
                s, n = evaluate_set(new_set)
                
                if s > current_score:
                    current_score = s
                    current_n = n
                    current_set = new_set
                    improved = True
                    break
            # If improved, restart while loop to re-generate potential swaps
            # If not improved, we break out of while loop? 
            # No, we should keep trying different samples if no improvement found in this sample?
            # But if we don't find improvement in 100 random swaps, we might be in a local optimum or flat region.
            # It's better to break and let the outer random restart handle it.
            # Or, we can try a few more samples.
            # Let's break to save time for restarts.
            
        return current_set, current_score, current_n
