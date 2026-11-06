import time
import random

N = 9
TOTAL_VERTS = 1 << N

# Precompute neighbor masks
neighbor_masks = [0] * TOTAL_VERTS
for v in range(TOTAL_VERTS):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    neighbor_masks[v] = m

def construct():
    # Beam Search parameters
    beam_width = 500
    max_steps = 300 # Safety cap, max possible is 512
    
    # State: (path, used_mask, non_last_exclusion)
    # We store path as a list of ints.
    initial_state = ([0], 1, 0)
    beam = [initial_state]
    
    best_path = [0]
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > 100: # Leave 20s for finishing
            break
            
        next_beam = []
        
        for path, used_mask, non_last_exclusion in beam:
            cur = path[-1]
            total_excluded = used_mask | non_last_exclusion
            candidates = neighbor_masks[cur] & ~total_excluded
            
            # Iterate through set bits in candidates
            c = candidates
            while c:
                lowest_bit = c & (-c)
                w = lowest_bit.bit_length() - 1
                c ^= lowest_bit
                
                new_path = path + [w]
                new_used_mask = used_mask | (1 << w)
                # cur becomes non-last
                new_non_last_exclusion = non_last_exclusion | neighbor_masks[cur]
                
                next_beam.append((new_path, new_used_mask, new_non_last_exclusion))
        
        if not next_beam:
            break
            
        # Select top beam_width states
        # Heuristic: Length is primary. 
        # To improve diversity, we might want to sort by length, then by some other metric.
        # For now, sort by length descending.
        # If there are many states with the same length, we pick arbitrarily or by some secondary score.
        # A good secondary score is the number of available moves from the new current vertex.
        
        def score(state):
            p, um, nle = state
            cur = p[-1]
            # Count available moves for next step
            cands = neighbor_masks[cur] & ~(um | nle)
            num_cands = bin(cands).count('1')
            return (len(p), num_cands)
            
        next_beam.sort(key=score, reverse=True)
        beam = next_beam[:beam_width]
        
        # Update best path
        if len(beam[0][0]) > len(best_path):
            best_path = beam[0][0]
            
        # Optimization: If all states in the beam have the same length and no more extensions are possible for any, we can stop?
        # Actually, if next_beam was empty, we broke.
        # If we just added states, we continue.
        
    return best_path
