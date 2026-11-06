import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Beam search parameters
    beam_width = 1000
    
    # Each beam is a tuple: (sequence, product_set, last_val, sum_val)
    # sequence: list of ints
    # product_set: set of all block products
    # last_val: last element in sequence
    # sum_val: sum of elements
    
    initial_beam = ([], set(), 1, 0)
    beams = [initial_beam]
    
    best_seq = []
    best_score = (0, 0) # (length, sum)
    
    while beams:
        if time.time() > deadline:
            break
            
        next_beams = []
        
        for seq, prods, last, s_val in beams:
            # Try all possible next values
            # Optimization: We can iterate from last+1 to 128
            for x in range(last + 1, 129):
                # Calculate new products formed by appending x
                # These are x, x*last, x*last*prev, ...
                # We can generate these efficiently
                
                # We need to check if any new product is in prods
                # New products are:
                # p_1 = x
                # p_2 = x * seq[-1]
                # p_3 = x * seq[-1] * seq[-2]
                # ...
                
                # Let's generate the list of new products
                new_prods = []
                prod = x
                # Iterate backwards through the sequence
                for a in reversed(seq):
                    new_prods.append(prod)
                    prod *= a
                new_prods.append(prod) # The full product including x and all previous
                
                # Check for collisions
                # If any new product is in prods, this x is invalid
                # Using intersection for speed
                if prods.intersection(new_prods):
                    continue
                
                # If valid, create new state
                new_seq = seq + [x]
                new_prods_set = prods.union(new_prods)
                new_last = x
                new_sum = s_val + x
                
                next_beams.append((new_seq, new_prods_set, new_last, new_sum))
                
                # Update global best
                cur_len = len(new_seq)
                cur_score = (cur_len, new_sum)
                if cur_score > best_score:
                    best_score = cur_score
                    best_seq = new_seq
        
        if not next_beams:
            break
            
        # Sort next_beams by score (length, sum) descending
        # We want the "best" beams to survive
        # To sort tuples of (list, set, int, int) by (len, sum), we need a key
        next_beams.sort(key=lambda b: (len(b[0]), b[3]), reverse=True)
        
        # Keep top beam_width
        beams = next_beams[:beam_width]
        
    return best_seq
