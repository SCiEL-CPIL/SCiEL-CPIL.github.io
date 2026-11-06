import time

N = 9
TOTAL_VERTS = 1 << N

# Precompute neighbor masks as 512-bit integers
# neighbor_masks[v] has bit w set iff v and w differ in exactly one bit
neighbor_masks = [0] * TOTAL_VERTS
for v in range(TOTAL_VERTS):
    m = 0
    for b in range(N):
        w = v ^ (1 << b)
        m |= (1 << w)
    neighbor_masks[v] = m

def construct():
    beam_width = 5000
    max_steps = 200
    
    # State: (path_list, exclusion_mask)
    # exclusion_mask: single 512-bit int where bit v is set iff vertex v is excluded
    # This consolidates used vertices and non-last neighbor exclusions into one mask
    
    beam = [([0], 1)]  # Start at vertex 0, exclude vertex 0
    
    best_path = [0]
    best_len = 1
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > 105:
            break
            
        next_beam = []
        
        for path, excl in beam:
            cur = path[-1]
            # Bit-parallel: get all valid next moves in one AND operation
            valid = neighbor_masks[cur] & ~excl
            
            c = valid
            while c:
                lb = c & (-c)
                w = lb.bit_length() - 1
                c ^= lb
                
                # New exclusion: add w, and all neighbors of cur (cur is no longer last)
                new_excl = excl | (1 << w) | neighbor_masks[cur]
                new_path = path + [w]
                
                next_beam.append((new_path, new_excl))
        
        if not next_beam:
            break
        
        def score(state):
            path, excl = state
            cur = path[-1]
            valid = neighbor_masks[cur] & ~excl
            return (len(path), bin(valid).count('1'))
        
        next_beam.sort(key=score, reverse=True)
        beam = next_beam[:beam_width]
        
        if len(beam[0][0]) > best_len:
            best_len = len(beam[0][0])
            best_path = list(beam[0][0])
    
    return best_path
