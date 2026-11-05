    def local_search(A_set, time_limit):
        """Fast local search with incremental count updates."""
        A_list = list(A_set)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        # Initial count calculation
        for i, x in enumerate(A_list):
            for y in A_list[i:]:
                s = x + y
                if min_sum <= s <= max_sum:
                    counts[s] += 1
        
        current_cost = get_cost_from_counts(counts)
        best_cost = current_cost
        best_A = set(A_set)
        
        def get_cost_from_counts(c):
            tc = c[min_sum:max_sum + 1]
            if np.any(tc == 0):
                return 10000 + int(np.sum(tc == 0))
            return int(np.max(tc))

        def add_element(x, A_list_ref, counts_ref):
            # x is not in A_list_ref
            for y in A_list_ref:
                s = x + y
                if min_sum <= s <= max_sum:
                    counts_ref[s] += 1
            A_list_ref.append(x)

        def remove_element(x, A_list_ref, counts_ref):
            # x is in A_list_ref
            idx = A_list_ref.index(x)
            # Remove from end for efficiency? Or just swap.
            # For correctness of indices, let's swap with last
            A_list_ref[idx] = A_list_ref[-1]
            A_list_ref.pop()
            for y in A_list_ref:
                s = x + y
                if min_sum <= s <= max_sum:
                    counts_ref[s] -= 1

        while time.time() < time_limit:
            if np.any(counts[min_sum:max_sum + 1] == 0):
                # Fix coverage
                # Find uncovered sums
                uncovered_mask = (counts[min_sum:max_sum + 1] == 0)
                uncovered_sums = [min_sum + i for i in range(len(uncovered_mask)) if uncovered_mask[i]]
                
                best_x = None
                best_gain = -1
                # Sample candidates to speed up? Or iterate all?
                # Iterating all 129 elements is fine.
                for x in all_elements - A_set:
                    gain = 0
                    for y in A_list:
                        s = x + y
                        if s in uncovered_sums:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is not None and best_gain > 0:
                    A_set.add(best_x)
                    add_element(best_x, A_list, counts)
                    current_cost = get_cost_from_counts(counts)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_set)
                else:
                    # Stuck, random add
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x = random.choice(candidates)
                        A_set.add(x)
                        add_element(x, A_list, counts)
                        current_cost = get_cost_from_counts(counts)
                continue

            peak = int(np.max(counts[min_sum:max_sum + 1]))
            if peak <= 2:
                break
            
            # Identify hot elements
            hot_elements = set()
            for s in target_sums:
                if counts[s] == peak:
                    for x in A_set:
                        y = s - x
                        if y in A_set and x <= y:
                            hot_elements.add(x)
                            if y != x:
                                hot_elements.add(y)
            
            candidates_to_remove = list(hot_elements)
            random.shuffle(candidates_to_remove)
            
            improved = False
            for x_out in candidates_to_remove:
                if time.time() >= time_limit:
                    break
                
                # Try replacing x_out
                candidates_to_add = list(all_elements - A_set)
                random.shuffle(candidates_to_add)
                
                for x_in in candidates_to_add[:15]:
                    # Simulate swap
                    # Remove x_out
                    idx = A_list.index(x_out)
                    A_list[idx] = A_list[-1]
                    last = A_list.pop()
                    
                    for y in A_list:
                        s = x_out + y
                        if min_sum <= s <= max_sum:
                            counts[s] -= 1
                    # Note: if last == x_out, we already popped it. 
                    # Wait, if x_out was last, A_list.pop() returns x_out.
                    # The loop above iterates over the NEW A_list (without x_out).
                    # This is correct.
                    
                    # Add x_in
                    for y in A_list:
                        s = x_in + y
                        if min_sum <= s <= max_sum:
                            counts[s] += 1
                    A_list.append(x_in)
                    
                    # Update set
                    A_set.discard(x_out)
                    A_set.add(x_in)
                    
                    new_cost = get_cost_from_counts(counts)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A_set)
                        break
                    else:
                        # Undo swap
                        # Remove x_in
                        A_list.pop() # x_in is last
                        for y in A_list:
                            s = x_in + y
                            if min_sum <= s <= max_sum:
                                counts[s] -= 1
                        
                        # Add back x_out
                        A_list.append(x_out)
                        for y in A_list[:-1]: # Exclude x_out itself
                            s = x_out + y
                            if min_sum <= s <= max_sum:
                                counts[s] += 1
                        # Wait, if x_out was the one swapped out, and we added it back to end.
                        # The pairs involving x_out are (x_out, y) for y in A_list[:-1].
                        # This is correct.
                        
                        A_set.discard(x_in)
                        A_set.add(x_out)
                
                if improved:
                    break
            
            if not improved:
                # Random perturbation
                if A_set:
                    x_out = random.choice(list(A_set))
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x_in = random.choice(candidates)
                        
                        # Perform swap
                        idx = A_list.index(x_out)
                        A_list[idx] = A_list[-1]
                        A_list.pop()
                        for y in A_list:
                            s = x_out + y
                            if min_sum <= s <= max_sum:
                                counts[s] -= 1
                        
                        A_list.append(x_in)
                        for y in A_list[:-1]:
                            s = x_in + y
                            if min_sum <= s <= max_sum:
                                counts[s] += 1
                            
                        A_set.discard(x_out)
                        A_set.add(x_in)
                        
                        new_cost = get_cost_from_counts(counts)
                        if new_cost <= current_cost:
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)
                        else:
                            # Undo
                            A_list.pop()
                            for y in A_list:
                                s = x_in + y
                                if min_sum <= s <= max_sum:
                                    counts[s] -= 1
                            
                            A_list.append(x_out)
                            for y in A_list[:-1]:
                                s = x_out + y
                                if min_sum <= s <= max_sum:
                                    counts[s] += 1
                            
                            A_set.discard(x_in)
                            A_set.add(x_out)

        return best_A, best_cost
