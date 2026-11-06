import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """Fast greedy growth to initialize."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    """
    Optimized HP Folding using Kink and Pivot moves with Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    # Global deadline: 320s to be safe against 330s kill
    global_deadline = time.time() + 320.0
    
    rng = random.Random(12345 + len(sequence))
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0
    
    # We will try multiple restarts.
    # For short sequences, many restarts. For long, fewer.
    # Let's allocate time per restart dynamically.
    
    while time.time() < global_deadline - 1.0:
        # Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: straight line
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
        # State
        pos = chain[:]
        pos_to_idx = {p: i for i, p in enumerate(pos)}
        
        # Calculate initial energy and contact count
        contacts = 0
        for i, p in enumerate(pos):
            if is_h[i]:
                for q in _neighbours(p):
                    j = pos_to_idx.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contacts += 1
        current_energy = -contacts
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = pos[:]
        
        # SA Parameters
        # Time slice for this restart
        # If we are early, take small slices. If late, take big slices?
        # Or just run until global deadline?
        # Better: Run for a fixed number of steps or time.
        # Let's run for 1 second per restart, or until global deadline.
        restart_deadline = min(global_deadline - 0.5, time.time() + 1.0)
        
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        steps_in_restart = 0
        
        while time.time() < restart_deadline:
            steps_in_restart += 1
            
            # Choose move type
            # Kinks are faster and more frequent
            if rng.random() < 0.7:
                # Kink Move
                if n < 3: break
                i = rng.randint(1, n - 2)
                p_prev = pos[i-1]
                p_curr = pos[i]
                p_next = pos[i+1]
                
                # Check if it's a valid kink geometry
                # New pos for i: p_prev + p_next - p_curr
                new_p_curr = tuple(p_prev[k] + p_next[k] - p_curr[k] for k in range(dimension))
                
                if new_p_curr == p_curr:
                    continue # No move
                
                # Check collision
                # new_p_curr must not be occupied by any index j != i
                # Note: p_prev and p_next are neighbors of i, so they are occupied by i-1 and i+1.
                # If new_p_curr == p_prev or new_p_curr == p_next, it's a collision with immediate neighbors?
                # Actually, if new_p_curr == p_prev, then bond length becomes 0. Invalid.
                # If new_p_curr == p_next, bond length 0. Invalid.
                
                if new_p_curr in pos_to_idx:
                    occupant = pos_to_idx[new_p_curr]
                    if occupant != i:
                        # Collision with another residue
                        continue
                
                # Calculate Energy Delta
                # Remove contacts for old p_curr
                # Add contacts for new p_curr
                
                delta_contacts = 0
                
                # Remove old
                if is_h[i]:
                    for q in _neighbours(p_curr):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i:
                            # Check if this contact was counted
                            # We count pair (i, j) if i < j? 
                            # In _energy, we counted if j > i+1.
                            # So we need to be careful with double counting.
                            # My _energy loop: for i, for q in neigh(p_i), if j > i+1.
                            # So contact (i, j) is counted once, at index i.
                            # Here, we are changing pos[i].
                            # Contacts involving i:
                            # 1. i is the 'first' index in pair (i, j) where j > i+1.
                            # 2. i is the 'second' index in pair (k, i) where k < i-1.
                            
                            # Case 1: i is first. j > i+1.
                            if j > i + 1 and is_h[j]:
                                delta_contacts -= 1 # Remove this contact
                            
                            # Case 2: i is second. k < i-1.
                            # The contact was counted at index k.
                            # So we need to check neighbors of p_curr to see if any k < i-1 is H.
                            # Wait, the contact is between pos[k] and pos[i].
                            # If pos[i] moves, the contact is broken.
                            # Was it counted? Yes, at index k.
                            if j < i - 1 and is_h[j]:
                                delta_contacts -= 1 # Remove this contact
                
                # Add new
                if is_h[i]:
                    for q in _neighbours(new_p_curr):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i:
                            if j > i + 1 and is_h[j]:
                                delta_contacts += 1
                            if j < i - 1 and is_h[j]:
                                delta_contacts += 1
                
                delta_energy = -delta_contacts
                
                # Acceptance
                accept = False
                if delta_energy < 0:
                    accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta_energy / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    pos_to_idx.pop(p_curr)
                    pos_to_idx[new_p_curr] = i
                    pos[i] = new_p_curr
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = pos[:]
                else:
                    # Reject, do nothing
                
                T *= cooling_rate

            else:
                # Pivot Move
                if n < 2: break
                k = rng.randint(1, n - 1)
                pivot_site = pos[k-1]
                
                # Extract segment k..n-1
                segment = pos[k:]
                seg_len = len(segment)
                
                # Remove from pos_to_idx
                for p in segment:
                    del pos_to_idx[p]
                
                # Choose rotation
                axis = rng.randint(0, dimension-1)
                direction = rng.choice([-1, 1])
                
                new_segment = []
                valid = True
                for p in segment:
                    v = [p[ax] - pivot_site[ax] for ax in range(dimension)]
                    if axis == 0:
                        if direction == 1: v_new = [v[0], -v[2], v[1]]
                        else: v_new = [v[0], v[2], -v[1]]
                    elif axis == 1:
                        if direction == 1: v_new = [v[2], v[1], -v[0]]
                        else: v_new = [-v[2], v[1], v[0]]
                    else:
                        if direction == 1: v_new = [-v[1], v[0], v[2]]
                        else: v_new = [v[1], -v[0], v[2]]
                    
                    new_p = tuple(pivot_site[ax] + v_new[ax] for ax in range(dimension))
                    new_segment.append(new_p)
                    
                    if new_p in pos_to_idx:
                        valid = False
                        break
                
                if not valid:
                    # Restore
                    for p, idx in zip(segment, range(k, n)):
                        pos_to_idx[p] = idx
                    continue
                
                # Calculate Energy Delta
                # This is more expensive.
                # We need to recalculate contacts for the boundary.
                
                # Old boundary contacts:
                # Contacts between segment (k..n-1) and R (0..k-1).
                # And internal contacts within segment? No, internal is invariant.
                # Wait, internal contacts within segment are invariant under rotation?
                # Yes, relative positions are preserved.
                # So we only need to check contacts between new_segment and R.
                
                old_contacts = 0
                if is_h[k-1]: # Pivot is in R
                    # Check neighbors of pivot? No, pivot is fixed.
                    # Check neighbors of segment residues that touch R.
                    pass
                
                # Efficient way:
                # Remove all contacts involving indices >= k that are with indices < k.
                # Add all contacts involving indices >= k (new pos) that are with indices < k.
                
                # Let's just recount boundary contacts.
                # For each i in k..n-1:
                #   if is_h[i]:
                #     for q in neighbors(pos[i]):
                #       if q in pos_to_idx (which is R now):
                #         j = pos_to_idx[q]
                #         if is_h[j]:
                #           # Check if this pair was counted in old_energy
                #           # We need to know if (i, j) was a contact in OLD state.
                #           # And if (i, j) is a contact in NEW state.
                
                # This requires knowing old positions.
                # We have `segment` (old) and `new_segment`.
                
                # Let's compute old_boundary_contacts and new_boundary_contacts.
                
                # Helper to count contacts between a list of positions (with indices) and R (pos_to_idx)
                def count_boundary_contacts(positions, start_idx):
                    c = 0
                    for offset, p in enumerate(positions):
                        idx = start_idx + offset
                        if is_h[idx]:
                            for q in _neighbours(p):
                                j = pos_to_idx.get(q)
                                if j is not None:
                                    if is_h[j]:
                                        # Count pair (idx, j)
                                        # To avoid double counting, we only count if idx < j?
                                        # No, idx >= k and j < k. So idx > j always.
                                        # So we count it once.
                                        c += 1
                    return c
                
                # Note: pos_to_idx currently contains ONLY R.
                old_boundary = count_boundary_contacts(segment, k)
                new_boundary = count_boundary_contacts(new_segment, k)
                
                delta_contacts = new_boundary - old_boundary
                delta_energy = -delta_contacts
                
                accept = False
                if delta_energy < 0:
                    accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta_energy / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    for p, idx in zip(new_segment, range(k, n)):
                        pos_to_idx[p] = idx
                        pos[idx] = p
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = pos[:]
                else:
                    for p, idx in zip(segment, range(k, n)):
                        pos_to_idx[p] = idx
                        # pos[idx] is already correct (old)
                
                T *= cooling_rate

    # Final hill climbing with Kinks only?
    # If time permits, do a quick pass of kinks on best_chain?
    # best_chain is a list of tuples.
    # Let's just return best_chain.
    
    if best_chain is None:
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
