<<<<<<< SEARCH
def _local_search_sa(sequence, dimension, chain, rng, time_limit=2.0):
    """
    Simulated Annealing local search with Pivot and Kink moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    start_time = time.time()
    
    # Precompute occupied set for quick lookups? 
    # We need to update it on every move.
    # For N=136, set operations are fast enough.
    
    while time.time() - start_time < time_limit:
        # Cooldown
        elapsed = time.time() - start_time
        T = max(T_end, T_start - (elapsed / time_limit) * (T_start - T_end))
        
        move_type = rng.random()
        
        valid_move = False
        new_chain = None
        
        if move_type < 0.6:
            # Pivot move: Rotate segment [i+1, j] around bond (i, i+1)
            if n < 5:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            p0 = current_chain[i]
            old_i1 = current_chain[i+1]
            
            # Find valid new positions for i+1 (neighbors of p0, not old_i1)
            valid_new_i1 = []
            for q in _neighbours(p0):
                if q != old_i1:
                    valid_new_i1.append(q)
            
            if not valid_new_i1:
                continue
            
            new_i1 = rng.choice(valid_new_i1)
            
            # Compute rotation: map vector (old_i1 - p0) to (new_i1 - p0)
            v_old = (old_i1[0] - p0[0], old_i1[1] - p0[1], old_i1[2] - p0[2])
            v_new = (new_i1[0] - p0[0], new_i1[1] - p0[1], new_i1[2] - p0[2])
            
            # Build rotation matrix from v_old to v_new
            # Both are unit vectors in 3D
            # Use cross product to find rotation axis
            cross = (v_old[1]*v_new[2] - v_old[2]*v_new[1],
                     v_old[2]*v_new[0] - v_old[0]*v_new[2],
                     v_old[0]*v_new[1] - v_old[1]*v_new[0])
            
            dot = v_old[0]*v_new[0] + v_old[1]*v_new[1] + v_old[2]*v_new[2]
            
            # If vectors are parallel or anti-parallel, handle separately
            if abs(dot - 1.0) < 1e-10:
                continue  # No rotation needed
            elif abs(dot + 1.0) < 1e-10:
                # 180 degree rotation - pick arbitrary perpendicular axis
                # Find a vector not parallel to v_old
                if abs(v_old[0]) < 0.9:
                    perp = (1, 0, 0)
                elif abs(v_old[1]) < 0.9:
                    perp = (0, 1, 0)
                else:
                    perp = (0, 0, 1)
                # Rotate 180 around perp
                def rot180(v, axis):
                    d = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]
                    return (2*axis[0]*d - v[0], 2*axis[1]*d - v[1], 2*axis[2]*d - v[2])
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rot180(v, perp)
                    new_positions.append((p0[0] + rv[0], p0[1] + rv[1], p0[2] + rv[2]))
            else:
                # Rodrigues' rotation formula
                # sin_theta = |cross|, cos_theta = dot
                sin_theta = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
                cos_theta = dot
                # Normalize cross
                axis = (cross[0]/sin_theta, cross[1]/sin_theta, cross[2]/sin_theta)
                
                def rodrigues(v, axis, cos_t, sin_t):
                    dot_av = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
                    cross_av = (axis[1]*v[2] - axis[2]*v[1],
                               axis[2]*v[0] - axis[0]*v[2],
                               axis[0]*v[1] - axis[1]*v[0])
                    return (v[0]*cos_t + cross_av[0]*sin_t + axis[0]*dot_av*(1-cos_t),
                            v[1]*cos_t + cross_av[1]*sin_t + axis[1]*dot_av*(1-cos_t),
                            v[2]*cos_t + cross_av[2]*sin_t + axis[2]*dot_av*(1-cos_t))
                
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rodrigues(v, axis, cos_theta, sin_theta)
                    # Round to nearest integer
                    new_positions.append((p0[0] + int(round(rv[0])), p0[1] + int(round(rv[1])), p0[2] + int(round(rv[2]))))
            
            # Check validity of new positions
            new_chain = list(current_chain)
            for k in range(i+1, j+1):
                new_chain[k] = new_positions[k-(i+1)]
            
            # Verify self-avoidance
            occupied_rest = set(current_chain[:i+1]) | set(current_chain[j+1:])
            valid = True
            for k in range(i+1, j+1):
                if new_chain[k] in occupied_rest:
                    valid = False
                    break
                occupied_rest.add(new_chain[k])
            
            # Verify connectivity
            if valid:
                if sum(abs(new_chain[i+1][a] - new_chain[i][a]) for a in range(dimension)) != 1:
                    valid = False
                if valid and j + 1 < n:
                    if sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension)) != 1:
                        valid = False
            
            if valid:
                current_chain = new_chain
                current_energy = _energy(sequence, current_chain)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                continue
        
        # Segment Shift Move
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        
        direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        
        # Check validity
        new_chain = list(current_chain)
        valid = True
        
        # Occupied set for non-moved parts
        occupied_set = set()
        for k in range(n):
            if i <= k <= j:
                continue
            occupied_set.add(current_chain[k])
        
        # Calculate new positions
        new_positions = []
        for k in range(i, j + 1):
            old_pos = current_chain[k]
            new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
            new_positions.append(new_pos)
            
        # Check self-avoidance against rest
        for k in range(i, j + 1):
            if new_positions[k-i] in occupied_set:
                valid = False
                break
        
        if valid:
            # Check connectivity at boundaries
            pos_prev = current_chain[i-1]
            pos_i_new = new_positions[0]
            dist1 = sum(abs(pos_prev[a] - pos_i_new[a]) for a in range(dimension))
            
            pos_j_new = new_positions[-1]
            pos_next = current_chain[j+1]
            dist2 = sum(abs(pos_j_new[a] - pos_next[a]) for a in range(dimension))
            
            if dist1 != 1 or dist2 != 1:
                valid = False
        
        if valid:
            for k in range(i, j + 1):
                new_chain[k] = new_positions[k-i]
            valid_move = True
        
        if not valid_move:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            delta = new_energy - current_energy
            if T > 0 and rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
    
    return best_chain, best_energy
=======
def _local_search_sa(sequence, dimension, chain, rng, time_limit=2.0):
    """
    Simulated Annealing local search with Pivot, Kink, and Segment Shift moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule - use geometric cooling for better exploration
    T_start = 8.0
    T_end = 0.05
    start_time = time.time()
    
    # Track recent moves to avoid cycling
    recent_energies = []
    plateau_count = 0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        # Geometric cooling
        progress = min(1.0, elapsed / time_limit)
        T = T_start * (T_end / T_start) ** progress
        
        # Increase kink move probability as we cool
        kink_prob = 0.3 if T > 2.0 else 0.4
        pivot_prob = 0.4
        shift_prob = 1.0 - kink_prob - pivot_prob
        
        move_type = rng.random()
        
        if move_type < kink_prob:
            # Kink move: try to move a single interior residue
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            
            p_prev = current_chain[i-1]
            p_curr = current_chain[i]
            p_next = current_chain[i+1]
            
            # Find valid new positions for residue i
            valid_positions = []
            for q in _neighbours(p_curr):
                if q == p_prev or q == p_next:
                    continue
                # Check if q is occupied by other residues
                occupied = set(current_chain)
                occupied.discard(p_curr)
                if q not in occupied:
                    valid_positions.append(q)
            
            if not valid_positions:
                continue
            
            new_pos = rng.choice(valid_positions)
            
            # Check connectivity
            if sum(abs(new_pos[a] - p_prev[a]) for a in range(dimension)) != 1:
                continue
            if sum(abs(new_pos[a] - p_next[a]) for a in range(dimension)) != 1:
                continue
            
            new_chain = list(current_chain)
            new_chain[i] = new_pos
            new_energy = _energy(sequence, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                recent_energies.append(new_energy)
                if len(recent_energies) > 10:
                    recent_energies.pop(0)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                    plateau_count = 0
                else:
                    plateau_count += 1
            else:
                delta = new_energy - current_energy
                if T > 0.01 and rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    recent_energies.append(new_energy)
                    if len(recent_energies) > 10:
                        recent_energies.pop(0)
                    plateau_count += 1
        
        elif move_type < kink_prob + pivot_prob:
            # Pivot move
            if n < 5:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            p0 = current_chain[i]
            old_i1 = current_chain[i+1]
            
            valid_new_i1 = []
            for q in _neighbours(p0):
                if q != old_i1:
                    valid_new_i1.append(q)
            
            if not valid_new_i1:
                continue
            
            new_i1 = rng.choice(valid_new_i1)
            
            v_old = (old_i1[0] - p0[0], old_i1[1] - p0[1], old_i1[2] - p0[2])
            v_new = (new_i1[0] - p0[0], new_i1[1] - p0[1], new_i1[2] - p0[2])
            
            cross = (v_old[1]*v_new[2] - v_old[2]*v_new[1],
                     v_old[2]*v_new[0] - v_old[0]*v_new[2],
                     v_old[0]*v_new[1] - v_old[1]*v_new[0])
            
            dot = v_old[0]*v_new[0] + v_old[1]*v_new[1] + v_old[2]*v_new[2]
            
            if abs(dot - 1.0) < 1e-10:
                continue
            elif abs(dot + 1.0) < 1e-10:
                if abs(v_old[0]) < 0.9:
                    perp = (1, 0, 0)
                elif abs(v_old[1]) < 0.9:
                    perp = (0, 1, 0)
                else:
                    perp = (0, 0, 1)
                
                def rot180(v, axis):
                    d = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]
                    return (2*axis[0]*d - v[0], 2*axis[1]*d - v[1], 2*axis[2]*d - v[2])
                
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rot180(v, perp)
                    new_positions.append((p0[0] + rv[0], p0[1] + rv[1], p0[2] + rv[2]))
            else:
                sin_theta = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
                cos_theta = dot
                axis = (cross[0]/sin_theta, cross[1]/sin_theta, cross[2]/sin_theta)
                
                def rodrigues(v, axis, cos_t, sin_t):
                    dot_av = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
                    cross_av = (axis[1]*v[2] - axis[2]*v[1],
                               axis[2]*v[0] - axis[0]*v[2],
                               axis[0]*v[1] - axis[1]*v[0])
                    return (v[0]*cos_t + cross_av[0]*sin_t + axis[0]*dot_av*(1-cos_t),
                            v[1]*cos_t + cross_av[1]*sin_t + axis[1]*dot_av*(1-cos_t),
                            v[2]*cos_t + cross_av[2]*sin_t + axis[2]*dot_av*(1-cos_t))
                
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rodrigues(v, axis, cos_theta, sin_theta)
                    new_positions.append((p0[0] + int(round(rv[0])), p0[1] + int(round(rv[1])), p0[2] + int(round(rv[2]))))
            
            new_chain = list(current_chain)
            for k in range(i+1, j+1):
                new_chain[k] = new_positions[k-(i+1)]
            
            occupied_rest = set(current_chain[:i+1]) | set(current_chain[j+1:])
            valid = True
            for k in range(i+1, j+1):
                if new_chain[k] in occupied_rest:
                    valid = False
                    break
                occupied_rest.add(new_chain[k])
            
            if valid:
                if sum(abs(new_chain[i+1][a] - new_chain[i][a]) for a in range(dimension)) != 1:
                    valid = False
                if valid and j + 1 < n:
                    if sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension)) != 1:
                        valid = False
            
            if valid:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    recent_energies.append(new_energy)
                    if len(recent_energies) > 10:
                        recent_energies.pop(0)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                        plateau_count = 0
                    else:
                        plateau_count += 1
                else:
                    delta = new_energy - current_energy
                    if T > 0.01 and rng.random() < math.exp(-delta / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        recent_energies.append(new_energy)
                        if len(recent_energies) > 10:
                            recent_energies.pop(0)
                        plateau_count += 1
        
        else:
            # Segment Shift Move
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 2)
            
            direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
            
            new_chain = list(current_chain)
            valid = True
            
            occupied_set = set()
            for k in range(n):
                if i <= k <= j:
                    continue
                occupied_set.add(current_chain[k])
            
            new_positions = []
            for k in range(i, j + 1):
                old_pos = current_chain[k]
                new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
                new_positions.append(new_pos)
            
            for k in range(i, j + 1):
                if new_positions[k-i] in occupied_set:
                    valid = False
                    break
            
            if valid:
                pos_prev = current_chain[i-1]
                pos_i_new = new_positions[0]
                dist1 = sum(abs(pos_prev[a] - pos_i_new[a]) for a in range(dimension))
                
                pos_j_new = new_positions[-1]
                pos_next = current_chain[j+1]
                dist2 = sum(abs(pos_j_new[a] - pos_next[a]) for a in range(dimension))
                
                if dist1 != 1 or dist2 != 1:
                    valid = False
            
            if valid:
                for k in range(i, j + 1):
                    new_chain[k] = new_positions[k-i]
                
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    recent_energies.append(new_energy)
                    if len(recent_energies) > 10:
                        recent_energies.pop(0)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                        plateau_count = 0
                    else:
                        plateau_count += 1
                else:
                    delta = new_energy - current_energy
                    if T > 0.01 and rng.random() < math.exp(-delta / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        recent_energies.append(new_energy)
                        if len(recent_energies) > 10:
                            recent_energies.pop(0)
                        plateau_count += 1
    
    return best_chain, best_energy
>>>>>>> REPLACE

