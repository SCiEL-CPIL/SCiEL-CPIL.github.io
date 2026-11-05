<<<<<<< SEARCH
def _pivot_search(sequence, chain, max_iters=2000, rng=None, time_limit=None):
    """
    Monte Carlo search using pivot moves.
    A pivot move rotates the suffix of the chain (from pivot to end) around the pivot point.
    This guarantees that if the new suffix does not collide with the prefix, the chain remains
    a valid self-avoiding walk with unit steps.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Pick a pivot point (index of the residue before the one being rotated)
        # Pivot index k means we rotate the chain from k+1 to n-1 around chain[k]
        # k ranges from 0 to n-2
        k = rng.randint(0, n - 2)
        
        # The axis of rotation is the bond between k and k+1.
        # In 2D, we can rotate the suffix by 90, 180, or 270 degrees.
        
        # Get the vector from k to k+1
        v_x = current_chain[k+1][0] - current_chain[k][0]
        v_y = current_chain[k+1][1] - current_chain[k][1]
        
        # We choose one of the 3 non-identity rotations
        rot_choice = rng.choice([1, 2, 3]) # 1: 90deg, 2: 180deg, 3: 270deg
        
        if rot_choice == 1:
            new_dx, new_dy = -v_y, v_x
        elif rot_choice == 2:
            new_dx, new_dy = -v_x, -v_y
        else:
            new_dx, new_dy = v_y, -v_x
        
        # Apply the rotation to the entire suffix [k+1, ..., n-1]
        # Let P0 = current_chain[k]
        # For j from k+1 to n-1:
        #   new_P[j] = P0 + R * (old_P[j] - P0)
        
        # Optimization: 
        # 1. Compute new positions for suffix.
        # 2. Check if any new position is in the set of occupied positions of the prefix [0, ..., k-1].
        
        prefix_occupied = set(current_chain[:k]) # Positions 0 to k-1
        
        valid = True
        new_suffix = []
        
        # Pre-compute the rotation function based on rot_choice
        if rot_choice == 1:
            def rot(x, y): return (-y, x)
        elif rot_choice == 2:
            def rot(x, y): return (-x, -y)
        else:
            def rot(x, y): return (y, -x)
            
        px, py = current_chain[k]
        
        new_chain = current_chain[:k+1] # Copy prefix including pivot
        new_chain.extend([None] * (n - k - 1))
        
        # Calculate new positions for k+1 to n-1
        for j in range(k + 1, n):
            ox = current_chain[j][0] - px
            oy = current_chain[j][1] - py
            nx, ny = rot(ox, oy)
            new_pos = (px + nx, py + ny)
            
            # Check collision with prefix (0 to k-1)
            if new_pos in prefix_occupied:
                valid = False
                break
            
            new_chain[j] = new_pos
        
        if not valid:
            # Move rejected due to collision
            continue
        
        # Calculate new energy
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance criterion: Simulated Annealing
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        
    return best_chain, best_energy
=======
def _pivot_search(sequence, chain, max_iters=2000, rng=None, time_limit=None):
    """
    Monte Carlo search using pivot moves with incremental energy tracking.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Build position -> index map for fast lookup
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    # Track contacts: set of frozenset({i, j}) for each H-H contact
    contacts = set()
    for i, p in enumerate(current_chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add(frozenset((i, j)))
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.02
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    start_time = time.time()
    deadline = start_time + (time_limit if time_limit else max_iters * 0.001)
    
    # Rotation tables: for each rotation type, map (dx, dy) -> (new_dx, new_dy)
    # rot 1: 90deg CCW: (x,y) -> (-y,x)
    # rot 2: 180deg:    (x,y) -> (-x,-y)
    # rot 3: 270deg CCW: (x,y) -> (y,-x)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Bias toward shorter suffixes: pick k with probability proportional to 1/sqrt(n-k)
        # This makes shorter pivots (faster collision checks) more likely
        r = rng.random()
        # Use a simple bias: k = n-2 - int(r * (n-1) * 0.5) with some randomness
        # Actually, let's just use uniform but weight by checking shorter first
        k = rng.randint(0, n - 2)
        
        # Choose rotation
        rot_choice = rng.randint(1, 3)
        
        px, py = current_chain[k]
        
        # Compute new suffix positions
        new_positions = {}
        valid = True
        
        if rot_choice == 1:
            # 90deg CCW
            for j in range(k + 1, n):
                ox = current_chain[j][0] - px
                oy = current_chain[j][1] - py
                nx, ny = -oy, ox
                new_pos = (px + nx, py + ny)
                
                # Check collision with prefix (indices 0 to k-1)
                # Only need to check if new_pos is in prefix positions
                if j > 0 and new_pos in pos_to_idx:
                    existing_idx = pos_to_idx[new_pos]
                    if existing_idx <= k - 1:
                        valid = False
                        break
                new_positions[j] = new_pos
        elif rot_choice == 2:
            # 180deg
            for j in range(k + 1, n):
                ox = current_chain[j][0] - px
                oy = current_chain[j][1] - py
                nx, ny = -ox, -oy
                new_pos = (px + nx, py + ny)
                
                if j > 0 and new_pos in pos_to_idx:
                    existing_idx = pos_to_idx[new_pos]
                    if existing_idx <= k - 1:
                        valid = False
                        break
                new_positions[j] = new_pos
        else:
            # 270deg CCW
            for j in range(k + 1, n):
                ox = current_chain[j][0] - px
                oy = current_chain[j][1] - py
                nx, ny = oy, -ox
                new_pos = (px + nx, py + ny)
                
                if j > 0 and new_pos in pos_to_idx:
                    existing_idx = pos_to_idx[new_pos]
                    if existing_idx <= k - 1:
                        valid = False
                        break
                new_positions[j] = new_pos
        
        if not valid:
            continue
        
        # Calculate energy delta incrementally
        # Contacts that involve indices >= k+1 may change
        # We need to find which old contacts are broken and which new ones are formed
        
        # Old contacts involving suffix indices
        old_suffix_contacts = set()
        for c in contacts:
            i, j = tuple(c)
            if i >= k + 1 or j >= k + 1:
                old_suffix_contacts.add(c)
        
        # New contacts involving suffix indices
        new_suffix_contacts = set()
        
        # Build a temporary pos_to_idx for the new chain
        # For efficiency, we only need to check neighbors of H residues in the suffix
        # and see if they match H residues in the prefix or other suffix positions
        
        # Create a dict of new positions for suffix
        new_pos_map = dict(pos_to_idx)  # Copy prefix
        for j, p in new_positions.items():
            new_pos_map[p] = j
        
        # Find new contacts
        for j in range(k + 1, n):
            if sequence[j] != 'H':
                continue
            p = new_positions[j]
            for q in _neighbours(p):
                i = new_pos_map.get(q)
                if i is not None and i < j and sequence[i] == 'H' and abs(i - j) > 1:
                    # Contact between i and j
                    if i >= k + 1 or j >= k + 1:  # Always true since j >= k+1
                        new_suffix_contacts.add(frozenset((i, j)))
        
        # Contacts that remain: old_suffix_contacts that are also in new_suffix_contacts
        remaining = old_suffix_contacts & new_suffix_contacts
        broken = old_suffix_contacts - new_suffix_contacts
        formed = new_suffix_contacts - old_suffix_contacts
        
        delta_E = len(formed) - len(broken)
        
        # Acceptance criterion
        if delta_E < 0 or (delta_E == 0 and rng.random() < 0.5) or rng.random() < math.exp(-delta_E / T):
            # Apply the move
            for j, p in new_positions.items():
                # Remove old position from pos_to_idx
                old_p = current_chain[j]
                del pos_to_idx[old_p]
                # Add new position
                pos_to_idx[p] = j
                current_chain[j] = p
            
            # Update contacts
            contacts -= broken
            contacts |= formed
            
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
    
    return best_chain, best_energy
>>>>>>> REPLACE

