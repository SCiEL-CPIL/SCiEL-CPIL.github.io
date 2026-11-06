import time
import random

N = 9
MAX_VERT = 1 << N

# Precompute neighbor masks
neighbors_mask = [0] * MAX_VERT
for v in range(MAX_VERT):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    neighbors_mask[v] = mask

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_path = []
    best_len = 0
    
    # Beam Search Parameters
    beam_width = 200
    noise_fraction = 0.1
    
    # Initial state
    # path: [0]
    # used: 1 << 0
    # forbidden: used | neighbors(path[:-1]) -> 1 (since path[:-1] is empty)
    # Wait, forbidden prevents adding v if v in used OR v in neighbors(path[:-1]).
    # For path=[0], path[:-1] is []. So forbidden = used = 1.
    # Valid v: v not in 1.
    # Correct.
    
    states = [(0, 1, [0], 1, 1)] # (score, flexibility, path, used, forbidden)
    # Score is just length for now, we re-sort every step.
    
    while True:
        if time.time() > deadline:
            break
            
        next_states = []
        
        for _, _, path, used, forbidden in states:
            cur = path[-1]
            # Generate valid next vertices
            # Candidates are neighbors of cur
            # v is valid if (forbidden | (1<<v)) & (1<<v) == 0 ?
            # No, v is valid if (forbidden & (1<<v)) == 0.
            # Because forbidden contains used and neighbors of path[:-1].
            
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if (forbidden & (1 << v)) == 0:
                    # Check if v is actually not used?
                    # forbidden includes used. So yes.
                    
                    # Calculate new state
                    new_used = used | (1 << v)
                    # new forbidden = new_used | (old forbidden | neighbors(cur))
                    # Wait.
                    # Old forbidden = used | S_prev (S_prev = neighbors of path[:-1])
                    # New path = path + [v]
                    # New path[:-1] = path[:-1] + [cur]
                    # New S_prev' = S_prev | neighbors(cur)
                    # New forbidden' = new_used | S_prev'
                    # = (used | (1<<v)) | ( (forbidden - used) | neighbors(cur) )?
                    # Simpler:
                    # new_forbidden = (used | (1<<v)) | (forbidden | neighbors_mask[cur])
                    # Note: forbidden already contains used.
                    # So new_forbidden = (forbidden | neighbors_mask[cur]) | (1<<v)
                    
                    new_forbidden = (forbidden | neighbors_mask[cur]) | (1 << v)
                    new_path = path + [v]
                    new_len = len(new_path)
                    
                    # Calculate flexibility: number of valid next moves from v
                    # Valid u from v: u in neighbors(v) and (new_forbidden & (1<<u)) == 0
                    # Note: new_forbidden contains new_used and neighbors(new_path[:-1]).
                    # new_path[:-1] ends with cur.
                    # So new_forbidden prevents u if u is in new_used or u is neighbor of new_path[:-1].
                    # This is correct for the NEXT step.
                    
                    flex = 0
                    for b2 in range(N):
                        u = v ^ (1 << b2)
                        if (new_forbidden & (1 << u)) == 0:
                            flex += 1
                    
                    # Score: prioritize length, then flexibility
                    # We will sort by (new_len, flex)
                    next_states.append((new_len, flex, new_path, new_used, new_forbidden))
        
        if not next_states:
            break
            
        # Sort by length desc, then flexibility desc
        next_states.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Keep top beam_width
        # Add noise
        keep_count = beam_width
        if len(next_states) > keep_count:
            # Take top keep_count - noise
            noise_count = int(keep_count * noise_fraction)
            top_count = keep_count - noise_count
            
            selected = next_states[:top_count]
            
            # Randomly pick from the rest
            rest = next_states[top_count:]
            if rest:
                random_selected = random.sample(rest, min(noise_count, len(rest)))
                selected.extend(random_selected)
            
            states = selected
        else:
            states = next_states
            
        # Update best
        current_best_len = max(s[0] for s in states)
        if current_best_len > best_len:
            best_len = current_best_len
            # Find the state with max length (and max flex if tie)
            best_state = max(states, key=lambda x: (x[0], x[1]))
            best_path = best_state[2]
            
    return best_path
