import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # State: (sequence, set_of_products, list_of_suffix_products)
    # Initial state
    beam = [( [], set(), [1] )]
    best_seq = []
    
    BEAM_WIDTH = 1000
    
    while True:
        if time.time() > deadline:
            break
            
        candidates = []
        
        for seq, prods, suffixes in beam:
            last = seq[-1] if seq else 1
            
            # Pruning: if even adding all possible numbers can't beat best, skip
            # Max additional elements is roughly 128 - last
            if len(seq) + (128 - last) <= len(best_seq):
                continue
                
            # Generate extensions
            for x in range(last + 1, 129):
                # Check collision
                # New products: { x * s for s in suffixes }
                # If any of these is in prods, collision
                valid = True
                for s in suffixes:
                    if x * s in prods:
                        valid = False
                        break
                
                if valid:
                    new_seq = seq + [x]
                    new_prods = prods.union(x * s for s in suffixes)
                    new_suffixes = [1] + [x * s for s in suffixes]
                    
                    candidates.append((new_seq, new_prods, new_suffixes))
        
        if not candidates:
            break
            
        # Update best
        for c in candidates:
            if len(c[0]) > len(best_seq):
                best_seq = c[0]
                
        # Sort candidates
        # Primary: length descending
        # Secondary: last element ascending (smaller last element allows more future choices)
        candidates.sort(key=lambda item: (-len(item[0]), item[0][-1]))
        
        beam = candidates[:BEAM_WIDTH]
        
    return best_seq
