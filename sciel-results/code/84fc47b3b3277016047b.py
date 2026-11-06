import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _grow(sequence, dimension, rng):
    """
    Greedy growth with some randomization to avoid getting stuck.
    Returns a list of tuples representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score the site
            # 1. Immediate H-H contact gain
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        # Check if it's a chain neighbor (i-1)
                        if idx == i - 1:
                            continue
                        if sequence[idx] == 'H':
                            gain += 1
            
            # 2. Free space (to avoid trapping)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # 3. Random tie-breaker
            rand = rng.random()
            
            # Weighted score: Gain is most important, free space second
            score = gain * 10.0 + free * 0.1 + rand
            options.append((score, site))
            
        if not options:
            # Dead end, try to backtrack? For simplicity, return None or random valid
            # In a robust implementation, we'd backtrack. Here, we just fail this init.
            return None
            
        # Pick best
        best_site = max(options)[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total energy of a chain."""
    if not chain:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _count_boundary_contacts(seg, start_idx, is_h, occupied, rest_set):
    """Count H-H contacts between segment and rest of chain."""
    c = 0
    for i_off, pos in enumerate(seg):
        idx = start_idx + i_off
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            if nb in rest_set:
                nb_idx = occupied[nb]
                # Skip chain neighbor (idx-1)
                if nb_idx != idx - 1:
                    if is_h[nb_idx]:
                        c += 1
    return c

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Time budget per call
    # Total 300s for 26 sequences. 300/26 ~ 11.5s.
    # Use 10.5s to be safe.
    max_time = 10.5
    start_time = time.time()
    
    rng = random.Random(42 + len(sequence))
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0 # 0 is no contacts, energy is negative, so lower is better
    
    # Pre-compute neighbor offsets for speed
    # We'll just use _neighbours for clarity, it's fast enough for N<200
    
    # Try multiple restarts
    while True:
        if time.time() - start_time > max_time:
            break
            
        # 1. Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: Straight line
            chain = [(i, 0, 0) for i in range(n)]
            
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
            
        # 2. Simulated Annealing
        # Temperature schedule
        # Max energy diff is roughly N/2. Let's start T high enough to accept bad moves.
        # T_max = 5.0, T_min = 0.01
        T_max = 5.0
        T_min = 0.01
        
        # Run SA for a certain number of steps or time
        # Let's run until time slice for this restart is up
        # We don't know how many restarts we'll get, so just run until global time is up?
        # No, better to have a fixed number of steps per restart to ensure diversity.
        # But if we have time, more steps is better.
        # Let's do: If first restart, run longer. Subsequent restarts, shorter.
        
        # Estimate steps/sec. ~5000-10000 steps/sec in Python for N=100?
        # Let's just use a time check inside the loop.
        
        steps = 0
        while True:
            if time.time() - start_time > max_time * 0.95:
                break
            
            # Choose move type
            # 70% Pivot, 30% Move (Translation)
            move_type = rng.random()
            
            # Choose pivot index k (1 to n-1)
            k = rng.randint(1, n - 1)
            
            old_segment = chain[k:]
            old_positions = old_segment
            
            # Remove old segment from occupied
            for p in old_positions:
                del occupied[p]
            
            new_segment = None
            valid = True
            
            if move_type < 0.7:
                # PIVOT
                # Rotate segment k..n-1 around chain[k-1]
                pivot_site = chain[k-1]
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                
                new_segment = []
                for p in old_positions:
                    v = [p[i] - pivot_site[i] for i in range(dimension)]
                    # Rotate v by 90 deg
                    if axis == 0:
                        if direction == 1:
                            v_new = [v[0], -v[2], v[1]]
                        else:
                            v_new = [v[0], v[2], -v[1]]
                    elif axis == 1:
                        if direction == 1:
                            v_new = [v[2], v[1], -v[0]]
                        else:
                            v_new = [-v[2], v[1], v[0]]
                    else: # axis 2
                        if direction == 1:
                            v_new = [-v[1], v[0], v[2]]
                        else:
                            v_new = [v[1], -v[0], v[2]]
                    
                    new_p = tuple(pivot_site[i] + v_new[i] for i in range(dimension))
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            else:
                # MOVE (Translation)
                # Translate segment by unit vector
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                offset = [0, 0, 0]
                offset[axis] = direction
                
                new_segment = []
                for p in old_positions:
                    new_p = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            if not valid:
                # Restore old
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
                continue
            
            # Calculate Energy Delta
            # Energy is -contacts.
            # Delta E = -(New Contacts - Old Contacts)
            
            # Helper to count contacts between a list of positions (segment) and occupied (rest)
            def count_boundary_contacts(seg_positions, seg_start_idx):
                contacts = 0
                for i_offset, pos in enumerate(seg_positions):
                    idx = seg_start_idx + i_offset
                    if is_h[idx] == 0:
                        continue
                    for nb in _neighbours(pos):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            # Skip chain neighbor
                            if nb_idx == idx - 1:
                                continue
                            if is_h[nb_idx] == 1:
                                contacts += 1
                return contacts
            
            # Note: 'occupied' currently contains only R (0..k-1)
            old_contacts = count_boundary_contacts(old_positions, k)
            new_contacts = count_boundary_contacts(new_segment, k)
            
            delta_contacts = new_contacts - old_contacts
            delta_energy = -delta_contacts
            
            # Acceptance
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                # Temperature
                # Simple linear cooling based on steps? 
                # Or based on time?
                # Let's use a fixed T for this restart that decreases?
                # Actually, for short runs, a constant T or slowly decaying T is fine.
                # Let's use T = T_max * (1 - step / total_estimated_steps)
                # But we don't know total steps.
                # Let's just use a constant T that is moderate, e.g., 2.0, 
                # and rely on multiple restarts for exploration.
                # Or, decay T based on time elapsed in this restart.
                # Let's do: T = T_min + (T_max - T_min) * (1 - (time.time() - restart_start) / restart_duration)
                # This is complex. Let's just use a constant T = 2.0 for simplicity and robustness.
                # If we want to be smarter, we can lower T as we find better energies.
                T = 2.0 
                try:
                    prob = math.exp(-delta_energy / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                # Update state
                for p, idx in zip(new_segment, range(k, n)):
                    occupied[p] = idx
                    chain[idx] = p
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Restore old
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
            
            steps += 1
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
