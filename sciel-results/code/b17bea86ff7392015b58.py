import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    n = 16
    num_edges = 120
    # Map (i, j) to index 0..119
    # i < j
    edge_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_indices[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    edges = [
                        edge_indices[(a,b)],
                        edge_indices[(a,c)],
                        edge_indices[(a,d)],
                        edge_indices[(b,c)],
                        edge_indices[(b,d)],
                        edge_indices[(c,d)]
                    ]
                    k4s.append(edges)
                    
    # Initialize with random 16 colours
    colours = np.random.randint(0, 16, size=num_edges)
    
    def get_violations(colours):
        violations = []
        for k4 in k4s:
            c_set = set(colours[k4])
            if len(c_set) < 5:
                violations.append(k4)
        return violations
        
    violations = get_violations(colours)
    best_colours = colours.copy()
    best_num_colours = len(set(colours))
    
    while time.time() < deadline:
        if not violations:
            # Try to reduce colour count
            # Find colour with min usage
            counts = np.bincount(colours, minlength=16)
            min_count_idx = np.argmin(counts)
            if counts[min_count_idx] == 0:
                # Already unused, effectively 15 colours used
                # Check if we can map another colour to it? No, it's empty.
                # Just return if we are happy with 15? 
                # The task says "Minimize". 
                # If we have a valid 16-colouring that only uses 15 distinct colours, that's great.
                # But we should try to merge two used colours.
                pass
            
            # Pick two colours to merge
            # Try merging the least used colour into another
            used_colours = np.where(counts > 0)[0]
            if len(used_colours) <= 15:
                # We are using <= 15 colours. 
                # If valid, this is a good candidate.
                if len(set(colours)) < best_num_colours:
                    best_num_colours = len(set(colours))
                    best_colours = colours.copy()
                
                # Try to merge further if possible
                if len(used_colours) > 1:
                    # Pick least used
                    c_merge = used_colours[np.argmin(counts[used_colours])]
                    # Pick another used colour
                    other_colours = [c for c in used_colours if c != c_merge]
                    if other_colours:
                        c_target = random.choice(other_colours)
                        # Apply merge
                        new_colours = colours.copy()
                        new_colours[new_colours == c_merge] = c_target
                        # Check validity
                        if not get_violations(new_colours):
                            colours = new_colours
                            violations = []
                            continue
            
            # If we can't merge, just wait or restart?
            # Let's just break or continue checking
            # Actually, if valid, we can return
            # But we might find 14? Unlikely.
            # Let's just return the best found if valid and stable?
            # No, keep searching for a while to see if 15 is possible.
            # If we are at 16 colours valid, and can't merge, we are stuck at 16.
            # Let's just return 16 if valid.
            return list(colours)

        else:
            # Pick a random violation
            k4 = random.choice(violations)
            # Pick a random edge in it
            edge_idx = random.choice(k4)
            # Change colour
            old_c = colours[edge_idx]
            new_c = random.randint(0, 15)
            colours[edge_idx] = new_c
            
            # Update violations
            # Instead of full recompute, we can update only affected K4s
            # But full recompute is fast enough.
            violations = get_violations(colours)
            
            # Track best
            if not violations:
                # Valid!
                current_distinct = len(set(colours))
                if current_distinct < best_num_colours:
                    best_num_colours = current_distinct
                    best_colours = colours.copy()
                # Try to optimize further? 
                # For now, just return
                return list(colours)

    # Time out: return best valid found, or current if valid
    if not get_violations(colours):
        return list(colours)
    
    # If no valid found, return seed? 
    # But seed is 17. 
    # We should probably return the best valid one found.
    # If none found, return seed to be safe? 
    # The prompt says "invalid result... scores 0".
    # So we MUST return a valid result.
    # If local search fails, fall back to seed.
    
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
