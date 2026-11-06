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
    beam_width = 10000
    max_steps = 200
    
    # State: (path_list, exclusion_mask)
    # exclusion_mask: single 512-bit int where bit v is set iff vertex v is excluded
    
    # Start with a fixed prefix to break symmetry and reduce initial branching
    # 0 -> 1 is a valid start.
    # From 1, valid moves are neighbors of 1 not in {0, 1, N(0)}.
    # N(0) = {1, 2, 4, 8, 16, 32, 64, 128, 256}
    # N(1) = {0, 3, 5, 9, 17, 33, 65, 129, 257}
    # Valid from 1: {3, 5, 9, ...}
    # By symmetry, we can pick 3.
    # Path: [0, 1, 3]
    # Excl after 0->1: {0, 1} U N(0)
    # Excl after 1->3: {0, 1, 3} U N(0) U N(1)
    
    initial_path = [0, 1, 3]
    initial_excl = (1 << 0) | (1 << 1) | (1 << 3) | neighbor_masks[0] | neighbor_masks[1]
    
    beam = [(initial_path, initial_excl)]
    
    best_path = list(initial_path)
    best_len = len(initial_path)
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > 110:
            break
            
        next_beam = []
        seen_states = set()
        
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
                
                # Deduplicate based on (last_vertex, exclusion_mask)
                # If two paths reach the same last vertex with the same exclusion mask,
                # their future possibilities are identical. We keep the one that is shorter?
                # Actually, if they have the same exclusion mask, they have visited the same set of vertices?
                # Not necessarily. But the constraint "non-consecutive differ by >= 2" is captured by the exclusion mask logic.
                # If we deduplicate, we might lose a path that is "better" in some other way?
                # No, the state space is defined by (current, excluded).
                # So deduplication is safe and beneficial.
                
                key = (w, new_excl)
                if key not in seen_states:
                    seen_states.add(key)
                    new_path = path + [w]
                    next_beam.append((new_path, new_excl))
        
        if not next_beam:
            break
        
        # Score: prioritize states with more available moves (greedy)
        # Using bit_count is faster than bin().count()
        def score(state):
            path, excl = state
            cur = path[-1]
            valid = neighbor_masks[cur] & ~excl
            return valid.bit_count()
        
        next_beam.sort(key=score, reverse=True)
        beam = next_beam[:beam_width]
        
        if len(beam[0][0]) > best_len:
            best_len = len(beam[0][0])
            best_path = list(beam[0][0])
    
    return best_path
