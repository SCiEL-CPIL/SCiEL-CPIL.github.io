import time

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin
    
    # State: (sequence, set_of_products)
    # We start with empty sequence
    # But strictly increasing in [2, 128]
    # Let's seed the beam with all single elements? 
    # Or just start with empty and expand.
    
    # To maximize speed, we can prune heavily.
    # Beam width
    W = 1000
    
    current_beam = [([], set())]
    best_seq = []
    
    # We will iterate until no more extensions or timeout
    # Each iteration increases the length of the sequences in the beam
    
    while current_beam:
        if time.time() - start_time > timeout:
            break
            
        next_candidates = []
        
        for seq, prods in current_beam:
            last_val = seq[-1] if seq else 1
            
            # Try all x > last_val
            # Optimization: If we are deep, maybe we can't add many more?
            for x in range(last_val + 1, 129):
                # Calculate new products ending at x
                # These are x * (suffix products of seq)
                # Suffix products of seq: 1, a_n, a_n*a_{n-1}, ...
                # Let's compute them on the fly
                
                new_prods = []
                curr_prod = x
                new_prods.append(curr_prod)
                
                # Check collision with existing prods
                # If curr_prod in prods, invalid
                if curr_prod in prods:
                    continue # Early exit? No, need to check all? 
                    # Actually if one collides, the whole extension is invalid.
                    # But wait, do we need to check all? Yes, if ANY new product is in prods, it's invalid.
                    # So we can break early if we find a collision?
                    # Yes.
                    
                # We need to generate all new prods to check for collisions.
                # And also to add to the set.
                
                # Let's re-structure:
                # 1. Generate list of new products.
                # 2. Check if any is in prods.
                # 3. If not, add to next_candidates.
                
                # Let's do it efficiently.
                
                suffix_prod = 1
                # The new products are x * (1, a_n, a_n a_{n-1}, ...)
                # Let's just compute them.
                
                # Actually, we can just compute the list.
                # L is small.
                
                # Reset the check
                valid = True
                new_prod_list = []
                p = x
                new_prod_list.append(p)
                if p in prods:
                    valid = False
                else:
                    # Multiply by elements of seq in reverse
                    for a in reversed(seq):
                        p *= a
                        new_prod_list.append(p)
                        if p in prods:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                new_seq = seq + [x]
                # New set of products
                # We don't need to create the full set yet if we are just filtering?
                # But we need it for the next step.
                # Creating a new set: prods | set(new_prod_list)
                
                # Optimization: If we have many candidates, creating sets is expensive.
                # But W is 1000, and branching factor is ~50. 50000 set unions.
                # Sets are small (~100 elements). Should be fast.
                
                new_prods_set = prods.union(new_prod_list)
                
                next_candidates.append((new_seq, new_prods_set))
                
                # Check time
                if time.time() - start_time > timeout:
                    return best_seq if len(best_seq) > len(seq) else seq # Return best found so far? 
                    # Actually construct() should return a valid sequence. 
                    # If we timeout, we should return the best valid sequence we have.
        
        if not next_candidates:
            break
        
        # Sort candidates
        # We want to keep the most promising ones.
        # Heuristic: 
        # 1. Prefer longer sequences? (All are same length here)
        # 2. Prefer sequences that end with smaller numbers? (More room to grow)
        # 3. Prefer sequences with "less dense" products? (Hard to measure)
        
        # Let's sort by last element ascending.
        next_candidates.sort(key=lambda item: item[0][-1])
        
        # Keep top W
        current_beam = next_candidates[:W]
        
        # Update best_seq
        # All in current_beam have length len(current_beam[0][0])
        # If this length is greater than best_seq, update.
        if len(current_beam[0][0]) > len(best_seq):
            best_seq = current_beam[0][0]
            
    return best_seq
