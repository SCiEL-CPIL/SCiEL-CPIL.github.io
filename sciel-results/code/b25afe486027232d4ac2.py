<<<<<<< SEARCH
    def local_search(codewords):
        """
        Aggressive local search:
        1. Remove redundant codewords.
        2. Try 1-swaps (remove 1, add 1) to enable further removals.
        3. Try 2-removals/1-additions (remove 2, add 1) to reduce size by 1.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant codewords
            counts = compute_coverage_counts(codewords)
            removed = False
            for c in list(codewords):
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    codewords.remove(c)
                    removed = True
                    improved = True
            if removed:
                continue # Restart loop with new counts

            # Phase 2: Try 2-removals / 1-additions (Reduce size by 1)
            # This is the key to going from 81 -> 73
            codeword_list = list(codewords)
            random.shuffle(codeword_list)
            found_swap = False
            for i in range(len(codeword_list)):
                if time.time() > deadline:
                    return codewords
                c1 = codeword_list[i]
                if c1 not in codewords: continue
                for j in range(i + 1, len(codeword_list)):
                    c2 = codeword_list[j]
                    if c2 not in codewords: continue
                    
                    # Tentatively remove c1 and c2
                    test_set = set(codewords)
                    test_set.remove(c1)
                    test_set.remove(c2)
                    
                    covered = get_coverage_mask(test_set)
                    uncovered = ((1 << total_words) - 1) & ~covered
                    if not uncovered:
                        # We can remove both? That would be a reduction by 2, which is even better.
                        # But usually we add 1. If we can remove 2 without adding, we do it.
                        codewords = test_set
                        improved = True
                        found_swap = True
                        break
                    
                    # Try to add one codeword to cover 'uncovered'
                    # We only care about candidates that cover at least one uncovered word
                    # Optimization: Only check candidates that are not in test_set
                    best_add = -1
                    best_gain = -1
                    # Iterate over all candidates? 729 is small.
                    for candidate in range(total_words):
                        if candidate in test_set:
                            continue
                        gain = (uncovered & neighbor_masks[candidate]).bit_count()
                        if gain == uncovered.bit_count():
                            # Perfect cover of the gap
                            best_add = candidate
                            best_gain = gain
                            break
                        elif gain > best_gain:
                            best_gain = gain
                            best_add = candidate
                    
                    if best_add != -1 and best_gain == uncovered.bit_count():
                        # Valid 2-swap
                        test_set.add(best_add)
                        # Verify full coverage just in case (should be guaranteed by logic above)
                        if is_valid(test_set):
                            codewords = test_set
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if found_swap:
                continue # Restart loop

            # Phase 3: Try 1-swaps (Remove 1, Add 1)
            # This doesn't reduce size immediately, but might allow a 2-swap or removal in the next iteration.
            # We look for a swap that maximizes the "slack" (minimum coverage count) to make future removals easier.
            # Or simply, try random swaps to escape local minima.
            if random.random() < 0.5: # Do this only sometimes to save time
                 c_to_remove = random.choice(list(codewords))
                 # Find best replacement
                 test_set = set(codewords)
                 test_set.remove(c_to_remove)
                 covered = get_coverage_mask(test_set)
                 uncovered = ((1 << total_words) - 1) & ~covered
                 
                 if not uncovered:
                     # This was a removal, handled in Phase 1. 
                     pass
                 else:
                     best_add = -1
                     best_score = -1
                     for candidate in range(total_words):
                         if candidate in test_set:
                             continue
                         gain = (uncovered & neighbor_masks[candidate]).bit_count()
                         if gain == uncovered.bit_count():
                             # Check if this swap improves "criticality"
                             # Calculate min coverage in new set
                             # This is expensive. Just accept the first valid swap?
                             # Or prefer swaps that keep the set size same but maybe "better" structure?
                             # For now, accept if it covers.
                             best_add = candidate
                             break
                     
                     if best_add != -1:
                         test_set.add(best_add)
                         if is_valid(test_set):
                             codewords = test_set
                             improved = True
                             break # Restart loop

        return codewords
=======
    def local_search(codewords):
        """
        Aggressive local search with fast bitmask operations:
        1. Remove redundant codewords using bitmask check.
        2. Try 2-removals/1-additions to reduce size by 1.
        3. Try 1-swaps to escape local minima.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant codewords (fast bitmask version)
            redundant = fast_redundancy_check(codewords)
            if redundant:
                for c in redundant:
                    codewords.remove(c)
                improved = True
                continue # Restart loop with new state

            # Phase 2: Try 2-removals / 1-additions (Reduce size by 1)
            codeword_list = list(codewords)
            random.shuffle(codeword_list)
            found_swap = False
            for i in range(len(codeword_list)):
                if time.time() > deadline:
                    return codewords
                c1 = codeword_list[i]
                if c1 not in codewords: continue
                for j in range(i + 1, len(codeword_list)):
                    c2 = codeword_list[j]
                    if c2 not in codewords: continue
                    
                    # Tentatively remove c1 and c2
                    test_set = set(codewords)
                    test_set.remove(c1)
                    test_set.remove(c2)
                    
                    covered = get_coverage_mask(test_set)
                    uncovered = ((1 << total_words) - 1) & ~covered
                    if not uncovered:
                        codewords = test_set
                        improved = True
                        found_swap = True
                        break
                    
                    # Try to add one codeword to cover 'uncovered'
                    best_add = -1
                    best_gain = -1
                    for candidate in range(total_words):
                        if candidate in test_set:
                            continue
                        gain = (uncovered & neighbor_masks[candidate]).bit_count()
                        if gain == uncovered.bit_count():
                            best_add = candidate
                            best_gain = gain
                            break
                        elif gain > best_gain:
                            best_gain = gain
                            best_add = candidate
                    
                    if best_add != -1 and best_gain == uncovered.bit_count():
                        test_set.add(best_add)
                        if is_valid(test_set):
                            codewords = test_set
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if found_swap:
                continue # Restart loop

            # Phase 3: Try 1-swaps to improve structure
            if random.random() < 0.7:
                # Try to find a swap that increases minimum coverage slack
                c_to_remove = random.choice(list(codewords))
                test_set = set(codewords)
                test_set.remove(c_to_remove)
                covered = get_coverage_mask(test_set)
                uncovered = ((1 << total_words) - 1) & ~covered
                
                if not uncovered:
                    # Shouldn't happen if Phase 1 worked, but handle it
                    codewords = test_set
                    improved = True
                    break
                else:
                    # Find candidate that maximizes minimum coverage in new configuration
                    best_add = -1
                    best_slack = -1
                    for candidate in range(total_words):
                        if candidate in test_set:
                            continue
                        if (uncovered & ~neighbor_masks[candidate]).bit_count() > 0:
                            # Can't fully cover the gap
                            continue
                        # This candidate covers all uncovered words
                        # Calculate resulting minimum coverage
                        new_counts = compute_coverage_counts(test_set | {candidate})
                        min_cov = min(new_counts[nb] for nb in neighbors[c_to_remove])
                        if min_cov > best_slack:
                            best_slack = min_cov
                            best_add = candidate
                        if best_slack >= 3:
                            break  # Good enough slack
                    
                    if best_add != -1:
                        test_set.add(best_add)
                        if is_valid(test_set):
                            codewords = test_set
                            improved = True
                            break

        return codewords
>>>>>>> REPLACE

