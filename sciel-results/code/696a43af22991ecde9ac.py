import time
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin before 60s kill
    
    # State: (sequence, set_of_all_products, list_of_suffix_products)
    # sequence: list of ints
    # set_of_all_products: set of products of all non-empty consecutive blocks
    # list_of_suffix_products: list of products of consecutive blocks ending at the last element
    
    initial_state = ([], set(), [])
    beam = [initial_state]
    best_seq = []
    
    # Beam width
    WIDTH = 500
    
    while beam:
        current_time = time.time()
        if current_time - start_time > time_limit:
            break
            
        next_beam = []
        
        for seq, all_prods, suffixes in beam:
            last_val = seq[-1] if seq else 1
            
            # Iterate through possible next values
            # Optimization: We can break early if we've generated too many? No, we need to evaluate all to prune.
            for x in range(last_val + 1, 129):
                # Calculate new suffixes
                # new_suffixes = [x] + [x * s for s in suffixes]
                new_suffixes = [x]
                for s in suffixes:
                    new_suffixes.append(x * s)
                
                # Check for collisions
                # If any new product is in all_prods, it's invalid
                # Note: new_suffixes might have duplicates if suffixes had duplicates? 
                # No, suffixes are distinct because the sequence is strictly increasing and >= 2.
                # Proof: Let seq be a1, ..., ak. Suffixes are ak, a(k-1)ak, ..., a1...ak.
                # Since ai >= 2, the products are strictly increasing in length?
                # P1 = ak
                # P2 = a(k-1)ak > ak
                # So yes, distinct.
                
                # Efficient intersection check
                # Convert to set for O(1) lookup? 
                # all_prods is already a set.
                # We just need to check if any element in new_suffixes is in all_prods.
                valid = True
                for p in new_suffixes:
                    if p in all_prods:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Create new state
                new_all_prods = all_prods.union(new_suffixes)
                new_seq = seq + [x]
                
                # Store state
                next_beam.append((new_seq, new_all_prods, new_suffixes))
                
                # Update best
                if len(new_seq) > len(best_seq):
                    best_seq = new_seq
        
        if not next_beam:
            break
            
        # Prune next_beam
        if len(next_beam) > WIDTH:
            # Sort by:
            # 1. Last element (ascending) - smaller is better
            # 2. Sum of elements (ascending) - smaller is better (keeps numbers small)
            # 3. Maybe length of all_prods? Smaller set is better?
            
            # The prompt mentioned "sum-based heuristic".
            # Let's sort by (last_element, sum(seq))
            next_beam.sort(key=lambda state: (state[0][-1], sum(state[0])))
            next_beam = next_beam[:WIDTH]
            
        beam = next_beam
        
    return best_seq
