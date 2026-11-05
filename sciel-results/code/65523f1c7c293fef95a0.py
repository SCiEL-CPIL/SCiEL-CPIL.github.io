import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # State: (sequence, product_set, suffix_products)
    initial_state = ([], set(), [])
    current_beam = [initial_state]
    best_seq = []
    best_len = 0
    
    BEAM_WIDTH = 5000
    
    while current_beam and time.time() < deadline:
        next_beam = []
        
        for seq, prods, suffixes in current_beam:
            last_val = seq[-1] if seq else 1
            if last_val >= 128:
                continue
            
            # Update best if this state is better
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
            for x in range(last_val + 1, 129):
                # Pruning: Can this branch possibly beat best_len?
                # Max length possible from x is 128 - x + 1 (x itself) ... 
                # Actually, if we pick x, the max length we can reach is len(seq) + 1 + (128 - x).
                # Because we can pick at most one number for each integer value > x up to 128.
                if len(seq) + 1 + (128 - x) <= best_len:
                    continue
                
                # Calculate new suffix products
                new_suffixes = [x]
                for s in suffixes:
                    new_suffixes.append(s * x)
                
                # Check for conflicts with existing products
                conflict = False
                for p in new_suffixes:
                    if p in prods:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                new_prods = prods.union(new_suffixes)
                new_seq = seq + [x]
                
                next_beam.append((new_seq, new_prods, new_suffixes))
        
        if not next_beam:
            break
            
        # Sort and prune
        # Primary: Length (desc)
        # Secondary: Last value (asc) - smaller allows more extensions
        next_beam.sort(key=lambda state: (-len(state[0]), state[0][-1] if state[0] else 0))
        
        current_beam = next_beam[:BEAM_WIDTH]
        
    return best_seq
