import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0 # 5 seconds margin
    
    # Beam search
    # State: (sequence, product_set, suffix_products)
    # suffix_products: list of products of suffixes of the sequence
    # e.g. seq [2, 3] -> suffixes: [3, 6]
    # When adding x, new suffixes: [x, x*3, x*6]
    
    beam = [([], set(), [])]
    
    best_seq = []
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for seq, prods, suffixes in beam:
            last_val = seq[-1] if seq else 1
            
            # Try adding x > last_val
            for x in range(last_val + 1, 129):
                # Check time
                if time.time() - start_time > timeout:
                    break
                
                # Compute new suffixes
                # new_suffixes[0] = x
                # new_suffixes[i] = x * suffixes[i-1]
                new_suffixes = [x]
                for s in suffixes:
                    new_suffixes.append(x * s)
                
                # Check for collisions
                # We need to check if any element in new_suffixes is in prods
                # Also, new_suffixes elements are distinct from each other because x > 1 and suffixes are distinct and increasing?
                # Suffixes are strictly decreasing? No, suffixes of [2,3] are 3, 6. Increasing.
                # So new_suffixes are x, 3x, 6x. Distinct.
                
                # Check intersection
                collision = False
                for p in new_suffixes:
                    if p in prods:
                        collision = True
                        break
                
                if collision:
                    continue
                
                # Valid extension
                new_prods = prods.union(new_suffixes)
                new_seq = seq + [x]
                
                next_beam.append((new_seq, new_prods, new_suffixes))
                
                # Update best
                if len(new_seq) > len(best_seq):
                    best_seq = new_seq
        
        if not next_beam:
            break
            
        # Prune beam to top K
        # Sort by length descending
        # Tie-breaker: sum of elements ascending? Or random?
        # Let's use sum ascending to keep "smaller" products?
        next_beam.sort(key=lambda x: (-len(x[0]), sum(x[0])))
        
        # Keep top 500
        beam = next_beam[:500]
        
        # If the longest sequence in the beam is the same as best_seq, 
        # and we can't extend any of them? 
        # The loop continues until next_beam is empty.
        
    return best_seq
