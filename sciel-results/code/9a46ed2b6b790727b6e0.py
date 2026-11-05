<<<<<<< SEARCH
def _try_move(chain, i, new_pos, occupied_set):
    """Try to move residue i to new_pos. Returns True if valid."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return False
    if new_pos in occupied_set:
        return False
    # Check adjacency to neighbors
    prev = chain[i-1]
    nxt = chain[i+1]
    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
        return False
    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
        return False
    return True

def _local_search(sequence, chain, max_iters=200, rng=None, time_limit=None):
    """Perform local search using single-residue moves and segment reversals."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        
        iters += 1
        improved = False
        
        # Strategy 1: Single residue moves
        num_tries = min(10, n - 2)
        for _ in range(num_tries):
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                occupied_set = set(chain)
                if not _try_move(chain, i, new_pos, occupied_set):
                    continue
                
                new_chain = list(chain)
                new_chain[i] = new_pos
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    chain = new_chain
                    improved = True
                    no_improve = 0
                    break
                elif new_energy < current_energy:
                    # Accept if better than current (hill climbing)
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        # Strategy 2: Segment reversal (for longer chains)
        if n >= 5 and rng.random() < 0.3:
            seg_len = rng.randint(3, min(7, n // 3))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len
            
            # Try to reverse the segment by checking if it's feasible
            # This is complex, so we'll try a simpler approach:
            # Pick two non-adjacent positions and try to swap their "direction"
            
            pass  # Skip complex segment moves for now
        
        if not improved:
            no_improve += 1
            if no_improve > 20:
                break
        else:
            no_improve = 0
    
    return best_chain, best_energy
=======
def _try_move(chain, i, new_pos, occupied_set):
    """Try to move residue i to new_pos. Returns True if valid."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return False
    if new_pos in occupied_set:
        return False
    # Check adjacency to neighbors
    prev = chain[i-1]
    nxt = chain[i+1]
    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
        return False
    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
        return False
    return True

def _segment_reversal(sequence, chain, rng, time_limit=None):
    """Attempt a 2-opt segment reversal move. Returns (new_chain, new_energy) or (None, None)."""
    n = len(chain)
    if n < 5:
        return None, None
    
    # Try to reverse a segment
    max_seg_len = min(8, n // 3)
    for _ in range(5):  # Try a few random segments
        seg_len = rng.randint(2, max_seg_len)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Check if reversal is feasible
        # After reversal, chain[start_idx] must be adjacent to chain[start_idx-1]
        # and chain[end_idx] must be adjacent to chain[end_idx+1]
        # Also, no self-avoidance violations in the reversed segment
        
        old_start = chain[start_idx]
        old_end = chain[end_idx]
        
        # After reversal:
        # new chain[start_idx] = old_end, must be adjacent to chain[start_idx-1]
        # new chain[end_idx] = old_start, must be adjacent to chain[end_idx+1]
        
        prev_node = chain[start_idx - 1]
        next_node = chain[end_idx + 1]
        
        if abs(old_end[0] - prev_node[0]) + abs(old_end[1] - prev_node[1]) != 1:
            continue
        if abs(old_start[0] - next_node[0]) + abs(old_start[1] - next_node[1]) != 1:
            continue
        
        # Check self-avoidance for the reversed segment
        # The reversed segment occupies the same set of positions, so self-avoidance is preserved
        # But we need to check that the endpoints connect properly and no other conflicts
        
        new_chain = list(chain)
        new_chain[start_idx:end_idx+1] = reversed(chain[start_idx:end_idx+1])
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify connectivity
        valid = True
        for k in range(n - 1):
            if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        return new_chain, new_energy
    
    return None, None

def _local_search(sequence, chain, max_iters=200, rng=None, time_limit=None):
    """Perform local search using single-residue moves and segment reversals."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        
        iters += 1
        improved = False
        
        # Build occupied set once per iteration for efficiency
        occupied_set = set(chain)
        
        # Strategy 1: Single residue moves
        num_tries = min(15, n - 2)
        for _ in range(num_tries):
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                if not _try_move(chain, i, new_pos, occupied_set):
                    continue
                
                new_chain = list(chain)
                new_chain[i] = new_pos
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    chain = new_chain
                    occupied_set = set(chain)
                    improved = True
                    no_improve = 0
                    break
                elif new_energy < current_energy:
                    # Accept if better than current (hill climbing)
                    chain = new_chain
                    current_energy = new_energy
                    occupied_set = set(chain)
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        # Strategy 2: Segment reversal (for longer chains)
        if n >= 5 and rng.random() < 0.4:
            new_chain, new_energy = _segment_reversal(sequence, chain, rng, time_limit)
            if new_chain is not None:
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    chain = new_chain
                    improved = True
                    no_improve = 0
                elif new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    no_improve = 0
        
        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
        else:
            no_improve = 0
    
    return best_chain, best_energy
>>>>>>> REPLACE

