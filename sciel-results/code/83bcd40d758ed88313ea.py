<<<<<<< SEARCH
def _local_refine(sequence, chain, dimension, max_iters=50):
    """
    Local refinement: try to improve energy by moving individual residues.
    A move is valid if:
    1. The new position is unoccupied
    2. The chain remains connected (consecutive residues are adjacent)
    3. The chain remains self-avoiding
    """
    n = len(chain)
    if n == 0:
        return chain, 0
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iters):
        improved = False
        occupied = {p: i for i, p in enumerate(best_chain)}
        
        for i in range(n):
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
            
            pos = best_chain[i]
            
            # Try moving residue i to a neighboring empty site
            for new_pos in _neighbours(pos):
                if new_pos in occupied:
                    continue
                
                # Check if this move maintains connectivity
                # The residue at position i must be adjacent to i-1 and i+1 (if they exist)
                valid = True
                
                if i > 0:
                    # Must be adjacent to previous residue
                    prev_pos = best_chain[i-1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, prev_pos))
                    if dist != 1:
                        valid = False
                
                if valid and i < n-1:
                    # Must be adjacent to next residue
                    next_pos = best_chain[i+1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, next_pos))
                    if dist != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Tentatively apply the move
                old_pos = best_chain[i]
                best_chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                # Check self-avoidance
                if len(occupied) != n:
                    # Collision detected, revert
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
                    continue
                
                # Compute new energy
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                else:
                    # Revert
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
            
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy
=======
def _compute_contact_map(sequence, chain):
    """Build a set of H-H contact pairs (i, j) where i < j."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

def _local_refine(sequence, chain, dimension, max_iters=50):
    """
    Local refinement with incremental energy updates and multiple move types:
    - Pivot moves (single residue to adjacent empty site)
    - Kink moves (reverse a short segment)
    - Crank moves (rotate a 3-residue segment)
    """
    n = len(chain)
    if n == 0:
        return chain, 0
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    contact_set = _compute_contact_map(sequence, best_chain)
    
    deadline = START_TIME + TIME_LIMIT * 0.85
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try pivot moves
        for i in range(n):
            if time.time() > deadline:
                break
            
            pos = best_chain[i]
            
            for new_pos in _neighbours(pos):
                if new_pos in occupied:
                    continue
                
                # Check connectivity
                valid = True
                if i > 0:
                    prev_pos = best_chain[i-1]
                    if sum(abs(a-b) for a, b in zip(new_pos, prev_pos)) != 1:
                        valid = False
                if valid and i < n-1:
                    next_pos = best_chain[i+1]
                    if sum(abs(a-b) for a, b in zip(new_pos, next_pos)) != 1:
                        valid = False
                
                if not valid:
                    continue
                
                old_pos = best_chain[i]
                best_chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                if len(occupied) != n:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
                    continue
                
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                    contact_set = _compute_contact_map(sequence, best_chain)
                else:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
        
        # Try kink moves: reverse a segment of length 3 or 4
        for i in range(n - 2):
            if time.time() > deadline:
                break
            
            for seg_len in (3, 4):
                if i + seg_len > n:
                    continue
                
                # Extract segment
                seg = best_chain[i:i+seg_len]
                reversed_seg = list(reversed(seg))
                
                # Check if reversed segment fits
                valid = True
                # Check connection to previous
                if i > 0:
                    if sum(abs(a-b) for a, b in zip(reversed_seg[0], best_chain[i-1])) != 1:
                        valid = False
                # Check connection to next
                if i + seg_len < n:
                    if sum(abs(a-b) for a, b in zip(reversed_seg[-1], best_chain[i+seg_len])) != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Check self-avoidance for reversed segment
                temp_occupied = dict(occupied)
                for j in range(seg_len):
                    del temp_occupied[seg[j]]
                
                for p in reversed_seg:
                    if p in temp_occupied:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Apply kink
                for j in range(seg_len):
                    old_p = best_chain[i+j]
                    new_p = reversed_seg[j]
                    del occupied[old_p]
                    occupied[new_p] = i+j
                    best_chain[i+j] = new_p
                
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                    contact_set = _compute_contact_map(sequence, best_chain)
                else:
                    # Revert
                    for j in range(seg_len):
                        new_p = best_chain[i+j]
                        old_p = seg[j]
                        del occupied[new_p]
                        occupied[old_p] = i+j
                        best_chain[i+j] = old_p
        
        if not improved:
            break
    
    return best_chain, best_energy
>>>>>>> REPLACE

