import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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

def _rotate_point(p, axis, angle):
    """
    Rotate point p around axis by 90, 180, or 270 degrees.
    axis is one of 'x', 'y', 'z'.
    angle is 1, 2, or 3 (representing 90, 180, 270).
    """
    x, y, z = p
    if axis == 'x':
        if angle == 1: return (x, -z, y)
        if angle == 2: return (x, -y, -z)
        if angle == 3: return (x, z, -y)
    elif axis == 'y':
        if angle == 1: return (z, y, -x)
        if angle == 2: return (-x, y, -z)
        if angle == 3: return (-z, y, x)
    elif axis == 'z':
        if angle == 1: return (y, -x, z)
        if angle == 2: return (-x, -y, z)
        if angle == 3: return (-y, x, z)
    return p

def _pivot_move(chain, i, angle, axis):
    """
    Perform a pivot move at index i.
    Returns new_chain if valid, else None.
    Also returns delta_energy.
    """
    # This function is a placeholder for the logic, 
    # but for speed, we will integrate the move generation and energy calculation 
    # into the SA loop to avoid creating new lists unnecessarily.
    pass

def _simulate_annealing(sequence, dimension, start_chain, time_limit):
    """
    Run Simulated Annealing starting from start_chain.
    Returns the best chain found.
    """
    N = len(sequence)
    if N < 2:
        return start_chain

    # Pre-compute H indices
    h_indices = [i for i, res in enumerate(sequence) if res == 'H']
    h_set = set(h_indices)

    current_chain = list(start_chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    initial_temp = 1.0
    cooling_rate = 0.999 # Adjust based on expected iterations
    
    rng = random.Random()
    
    # We need to determine the axis and angle for the pivot.
    # The pivot is at index i. The bond is between i-1 and i.
    # The direction of the bond determines the axis.
    
    while time.time() - start_time < time_limit:
        # Pick a random pivot index
        i = rng.randint(2, N - 1)
        
        # Determine the axis of the bond chain[i-1] -> chain[i]
        p_prev = current_chain[i-1]
        p_curr = current_chain[i]
        
        # Determine which axis is different
        if p_prev[0] != p_curr[0]:
            axis = 'x'
        elif p_prev[1] != p_curr[1]:
            axis = 'y'
        else:
            axis = 'z'
            
        # Pick a random angle (1, 2, or 3)
        angle = rng.randint(1, 3)
        
        # Generate new positions for the segment chain[i:]
        # Rotate relative to p_prev
        new_segment = [None] * (N - i)
        
        # The first point of the segment is chain[i]
        # Vector from p_prev to chain[i]
        v0 = tuple(p_curr[k] - p_prev[k] for k in range(dimension))
        
        # Rotate v0
        # Note: _rotate_point expects absolute coordinates? 
        # Let's modify _rotate_point to work with vectors or adjust here.
        # Easier: Translate segment to origin, rotate, translate back.
        
        valid = True
        new_occupied = dict(current_occupied) # Copy is expensive. 
        # Instead, we will check collisions against the fixed part only.
        
        # Fixed part is chain[:i]
        # We need to check if any new position in new_segment is in the fixed part's occupied set.
        
        # Let's construct the new segment coordinates
        for k in range(i, N):
            # Vector from p_prev to chain[k]
            v = tuple(current_chain[k][dim] - p_prev[dim] for dim in range(dimension))
            # Rotate v
            rv = _rotate_point(v, axis, angle)
            # New position
            new_p = tuple(p_prev[dim] + rv[dim] for dim in range(dimension))
            new_segment[k - i] = new_p
            
            # Check collision with fixed part (chain[:i])
            # We can use a set of fixed positions for O(1) lookup
            # But we don't have a separate set for fixed part.
            # We can check if new_p is in current_occupied and current_occupied[new_p] < i
            
            if new_p in current_occupied:
                if current_occupied[new_p] < i:
                    valid = False
                    break
                # If current_occupied[new_p] >= i, it's part of the moving segment, 
                # which is allowed (it's its own old position or another moving part's old position).
                # Wait, we are replacing the whole segment. 
                # So we must ensure new_p does not collide with FIXED part.
                # It can collide with its own old position? No, it's a new configuration.
                # It can collide with other parts of the moving segment? 
                # The rotation is a rigid body motion, so self-collisions within the segment are preserved.
                # Since the original segment was self-avoiding, the rotated one is too.
                # So we only need to check against the fixed part.
        
        if not valid:
            continue
            
        # Calculate Delta Energy
        # Old contacts between fixed and moved
        old_contacts = 0
        for k in range(i, N):
            if sequence[k] == 'H':
                p = current_chain[k]
                for q in _neighbours(p):
                    j = current_occupied.get(q)
                    if j is not None and j < i and sequence[j] == 'H':
                        # j is in fixed part, k is in moved part
                        # Ensure they are not adjacent in chain? 
                        # j < i and k >= i. 
                        # If k == i and j == i-1, they are adjacent. 
                        # The problem says "neighbouring lattice sites that are NOT adjacent along the chain".
                        # So we must exclude |j-k| == 1.
                        if abs(j - k) > 1:
                            old_contacts += 1
                            
        new_contacts = 0
        for k in range(i, N):
            if sequence[k] == 'H':
                p = new_segment[k - i]
                for q in _neighbours(p):
                    j = current_occupied.get(q)
                    if j is not None and j < i and sequence[j] == 'H':
                        if abs(j - k) > 1:
                            new_contacts += 1
                            
        delta_e = -(new_contacts - old_contacts)
        
        # Metropolis acceptance
        if delta_e < 0 or rng.random() < pow(2.71828, -delta_e / initial_temp):
            # Accept move
            # Update chain
            for k in range(i, N):
                current_chain[k] = new_segment[k - i]
                current_occupied[current_chain[k]] = k
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        # Cool down
        initial_temp *= cooling_rate
        if initial_temp < 0.01:
            initial_temp = 0.01
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths."""
    # Time budget per sequence
    # Total budget 240s / 26 sequences ~ 9.2s. 
    # Let's use 8s to be safe.
    time_limit = 8.0
    
    # Generate initial state
    rng = random.Random(12345 + len(sequence))
    start_chain = _grow(sequence, dimension, rng)
    
    if start_chain is None:
        # Fallback: straight line
        start_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    # Run SA
    best_chain = _simulate_annealing(sequence, dimension, start_chain, time_limit)
    
    return [list(p) for p in best_chain]
