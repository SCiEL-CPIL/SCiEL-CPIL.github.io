import time
from collections import defaultdict

def construct():
    """
    Finds a strictly increasing sequence of integers in [2, 128] maximizing length
    such that products of all non-empty consecutive blocks are distinct.
    Uses Beam Search.
    """
    start_time = time.time()
    TIME_LIMIT = 50.0  # Safety margin for 60s/90s limit
    
    # Beam search parameters
    BEAM_WIDTH = 2000
    
    # Initial state: empty sequence, empty product set
    # We start by trying each number 2..128 as the first element
    # State: (sequence_tuple, product_set, last_val)
    # To save memory and speed up checks, we store product_set as a Python set of ints.
    
    # Initial beam: all single elements
    # (val, {val}, val)
    beam = []
    for x in range(2, 129):
        beam.append(((x,), {x}, x))
        
    best_seq = ()
    best_len = 0
    
    while beam:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        next_beam = []
        
        for seq, prods, last_val in beam:
            # Check if this state is better than current best
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
                
            # Generate children
            # Next value must be > last_val and <= 128
            # Also, we can prune if max possible length from here cannot beat best_len
            # Max possible length = len(seq) + (128 - last_val)
            # If len(seq) + (128 - last_val) <= best_len, skip expansion?
            # Actually, if we can't even reach best_len, we don't need to keep it for optimization, 
            # but we might need it if we haven't found the true max yet.
            # However, if we found a sequence of length 10, and a state has len 5 and last_val 120,
            # max len is 5 + 8 = 13. It can beat 10.
            # If len 5 and last_val 125, max len is 5+3=8. It cannot beat 10. 
            # So we can prune states that cannot beat the current best.
            
            if len(seq) + (128 - last_val) <= best_len:
                continue
                
            for x in range(last_val + 1, 129):
                # Calculate new products
                # New products are x, x*last_suffix, x*last_2_suffix, ...
                # We don't have suffixes stored directly, but we can derive them?
                # No, it's easier to just compute the new products from the sequence.
                # Wait, computing from sequence is O(L^2). 
                # Storing suffix products in the state is better.
                # Let's change state to (seq, prods, suffixes)
                # suffixes = [last_elem, last2*last1, ...]
                
                # Let's refactor to include suffixes in the state for efficiency.
                pass # Pseudo code, will implement in final code

    # The loop above was pseudo. Let's implement properly.
    pass

def construct():
    start_time = time.time()
    TIME_LIMIT = 50.0
    BEAM_WIDTH = 1000 # Start with 1000, maybe increase if fast
    
    # State: (tuple_seq, set_prods, list_suffixes, last_val)
    # list_suffixes: [p_1, p_2, ..., p_k] where p_i is product of last i elements
    # Note: p_1 is last_val.
    
    beam = []
    for x in range(2, 129):
        beam.append(((x,), {x}, [x], x))
        
    best_seq = ()
    best_len = 0
    
    # Sort initial beam? All len 1. Prefer larger x? 
    # Larger x leaves less room but might have fewer small factor collisions.
    # Let's just keep all, they are all valid.
    
    while beam:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        next_beam = []
        
        for seq, prods, suffixes, last_val in beam:
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
                
            # Pruning: if we can't beat best_len, skip
            # Max extension count is 128 - last_val
            if len(seq) + (128 - last_val) <= best_len:
                continue
                
            for x in range(last_val + 1, 129):
                # Compute new suffixes
                # new_suffixes[0] = x
                # new_suffixes[i] = x * suffixes[i-1]
                
                new_suffixes = [x]
                # Check if x collides
                if x in prods:
                    continue
                
                # Check other suffixes
                valid = True
                # We can stop early if product gets huge, but max product is 128^15 which is big, Python handles it.
                # But we only need to check against prods.
                
                # Build new suffixes and check
                new_suffixes = [x]
                if x in prods:
                    continue
                    
                # Calculate rest
                # suffixes is [s_1, s_2, ...] where s_1 is last elem of seq
                # New suffixes: [x, x*s_1, x*s_2, ...]
                
                # We need to check if any new product is in prods
                # Also, we must add new products to prods for the child state
                
                new_prods = set(prods) # Copy is expensive. 
                # Optimization: We can check collisions first without copying.
                
                # Collision check:
                # 1. x not in prods
                # 2. x * s_i not in prods for all i
                
                # If valid, create child state.
                # To avoid copying large sets for invalid candidates, do check first.
                
                # Re-verify x
                if x in prods:
                    continue
                    
                # Check x * suffixes
                # Note: x * s_i could be very large, but that's fine.
                # Also, x * s_i might be the same as another new product?
                # No, since seq is strictly increasing, s_i are distinct and increasing.
                # x is constant. So x*s_i are distinct.
                # But we must check if they collide with OLD prods.
                
                collision = False
                new_suffix_list = [x]
                for s in suffixes:
                    ps = x * s
                    new_suffix_list.append(ps)
                    if ps in prods:
                        collision = True
                        break
                        
                if collision:
                    continue
                
                # Also need to check if new products collide with EACH OTHER?
                # No, x and x*s_i are all distinct because s_i >= 1 (actually s_i >= 2) and strictly increasing.
                # So x < x*s_1 < x*s_2 ...
                
                # If valid, create new state
                new_seq = seq + (x,)
                # We don't need to copy prods if we just store the set and update?
                # No, beam search requires independent states.
                # We must copy the set.
                # Optimization: Only add to set if we are sure it's valid (which we are).
                # But copying a set of size ~L^2 is O(L^2).
                
                new_prods = set(prods)
                new_prods.update(new_suffix_list)
                
                next_beam.append((new_seq, new_prods, new_suffix_list, x))
                
        # Update beam
        # Score next_beam
        # Primary: Length (descending)
        # Secondary: Sum (descending) - per knowledge
        # Tertiary: Min next available? Or just random/diversity?
        # Let's sort by (len, sum) descending.
        
        if not next_beam:
            break
            
        # To improve diversity, we might want to keep some low-sum states if they have different structures?
        # But knowledge says larger sum is better.
        
        next_beam.sort(key=lambda item: (len(item[0]), sum(item[0])), reverse=True)
        
        # Keep top BEAM_WIDTH
        beam = next_beam[:BEAM_WIDTH]
        
    return list(best_seq)
