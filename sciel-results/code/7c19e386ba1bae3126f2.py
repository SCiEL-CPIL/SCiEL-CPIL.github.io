import time

def construct():
    start_time = time.time()
    # Use 55s to be safe within the 60s limit
    deadline = start_time + 55.0
    
    best_seq = []
    best_len = 0
    best_prods = set()

    # Beam width
    BEAM_WIDTH = 100
    
    # State: (sequence, products_set)
    # We store products in a set for O(1) lookup
    # Initial state
    current_beam = [([], set())]
    
    # Helper to check if adding x to current_seq creates unique products
    # Returns the set of new products if valid, else None
    def try_add(current_seq, x, current_products):
        # Calculate new products: x, x*last, x*last*second_last, etc.
        new_prods = []
        prod = x
        new_prods.append(prod)
        
        # Calculate products of x * suffix of current_seq
        for a in reversed(current_seq):
            prod *= a
            new_prods.append(prod)
            
        # Check for collisions with existing products
        # Also check for collisions within new_prods? 
        # No, new_prods are distinct because the sequence is strictly increasing and >= 2.
        # e.g., seq=[2,3], x=4. New prods: 4, 4*3=12, 4*3*2=24. Distinct.
        # So we only need to check against current_products.
        for p in new_prods:
            if p in current_products:
                return None # Collision
                
        return new_prods

    while True:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for seq, prods in current_beam:
            last_picked = seq[-1] if seq else 1
            
            # Pruning: Max possible extension
            # Remaining numbers are last_picked+1 to 128.
            # Count is 128 - last_picked.
            # If current_len + remaining <= best_len, this beam cannot beat the best.
            # However, we might still want to keep it if it's the only one? 
            # No, if it can't beat best_len, it's useless for finding a longer seq.
            # But what if best_len is 0? Then we keep everything.
            if len(seq) + (128 - last_picked) <= best_len:
                continue

            if last_picked >= 128:
                continue

            # Try adding next numbers
            # We iterate x from last_picked + 1 to 128
            for x in range(last_picked + 1, 129):
                if time.time() > deadline:
                    break
                
                new_prods_list = try_add(seq, x, prods)
                
                if new_prods_list is not None:
                    # Create new state
                    new_seq = seq + [x]
                    new_prods = prods.union(new_prods_list)
                    
                    # Update best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                        best_prods = new_prods
                        
                    next_beam.append((new_seq, new_prods))
                    
                    # Early break if we've already found a very long sequence?
                    # No, we need to fill the beam.
        
        if not next_beam:
            break
            
        # Sort next_beam to keep the best candidates
        # Heuristic: 
        # 1. Longer sequences are better.
        # 2. For same length, smaller last element is better (more room).
        # 3. Tie-breaker: arbitrary or based on product set size? 
        #    Actually, smaller product set size is impossible (determined by length).
        #    Let's just use (len, -last_element) or similar.
        #    Wait, if lengths are equal, smaller last element allows more future extensions.
        #    So we sort by:
        #    - length (descending)
        #    - last element (ascending)
        
        # To avoid keeping duplicates (same seq), we could deduplicate, 
        # but the generation is unique per path, so same seq implies same path?
        # Not necessarily, different paths can lead to same seq? 
        # No, the seq IS the path. So no duplicates in next_beam from different parents 
        # unless we merge beams? We don't merge.
        
        # However, we might have multiple beams leading to the same sequence? 
        # No, each beam is a unique sequence.
        
        # Sort
        next_beam.sort(key=lambda item: (-len(item[0]), item[0][-1] if item[0] else 0))
        
        # Trim to beam width
        current_beam = next_beam[:BEAM_WIDTH]
        
        # Optimization: If the best possible length from the best beam is <= best_len, 
        # we can stop.
        # Best beam is current_beam[0].
        if current_beam:
            best_beam_seq = current_beam[0][0]
            max_possible = len(best_beam_seq) + (128 - (best_beam_seq[-1] if best_beam_seq else 1))
            if max_possible <= best_len:
                break

    return best_seq
