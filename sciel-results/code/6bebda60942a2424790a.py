import random
import time
import math

_START_TIME = time.time()
_GLOBAL_LIMIT = 325.0 # 5s safety margin before 330s kill

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
    
    # Check global deadline
    if time.time() - _START_TIME > _GLOBAL_LIMIT:
        # Return a valid straight line to satisfy self-avoidance and unit steps
        return [[i] + [0]*(dimension-1) for i in range(n)]
    
    # Calculate remaining time for this specific call
    # We want to leave a small buffer for the next calls
    remaining_global = _GLOBAL_LIMIT - (time.time() - _START_TIME)
    # If we have very little time, just do a quick greedy or straight line
    if remaining_global < 1.0:
        chain = _grow(sequence, dimension, random.Random(12345 + len(sequence)))
        if chain is None:
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        return [list(p) for p in chain]
    
    rng = random.Random(12345 + len(sequence))
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0
    
    # We will try multiple restarts within the remaining global time
    # But we must stop if global limit is reached.
    # Let's allocate a fraction of the remaining time to this sequence.
    # A simple heuristic: use up to 90% of remaining time, but cap it to avoid overshooting if this is the last sequence.
    # Actually, just use the global limit as the hard stop.
    
    while time.time() - _START_TIME < _GLOBAL_LIMIT - 0.5:
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
        # Run until global deadline or a local restart limit (e.g., 2s)
        restart_deadline = min(_GLOBAL_LIMIT - 0.5, time.time() + 2.0)
        
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        while time.time() - _START_TIME < _GLOBAL_LIMIT - 0.5:
            if time.time() > restart_deadline:
                break # Restart
            
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
                if new_p_curr in pos_to_idx:
                    occupant = pos_to_idx[new_p_curr]
                    if occupant != i:
                        continue
                
                # Calculate Energy Delta
                delta_contacts = 0
                
                # Remove old
                if is_h[i]:
                    for q in _neighbours(p_curr):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i:
                            if j > i + 1 and is_h[j]:
                                delta_contacts -= 1
                            if j < i - 1 and is_h[j]:
                                delta_contacts -= 1
                
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
                
                T *= cooling_rate

            else:
                # Pivot Move
                if n < 2: break
                k = rng.randint(1, n - 1)
                pivot_site = pos[k-1]
                
                # Extract segment k..n-1
                segment = pos[k:]
                
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
                def count_boundary_contacts(positions, start_idx):
                    c = 0
                    for offset, p in enumerate(positions):
                        idx = start_idx + offset
                        if is_h[idx]:
                            for q in _neighbours(p):
                                j = pos_to_idx.get(q)
                                if j is not None:
                                    if is_h[j]:
                                        c += 1
                    return c
                
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
                
                T *= cooling_rate

    if best_chain is None:
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
