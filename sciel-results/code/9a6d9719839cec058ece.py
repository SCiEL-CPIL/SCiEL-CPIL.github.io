import random
import time
import math

def _rotate_vector(v, axis, angle):
    """
    Rotate vector v (tuple of 3 ints) around axis (0, 1, 2) by angle (1, 2, 3).
    angle 1: 90 deg, angle 2: 180 deg, angle 3: 270 deg.
    """
    x, y, z = v
    if axis == 0: # Rotate around X
        if angle == 1: # 90 deg: (x, y, z) -> (x, -z, y)
            return (x, -z, y)
        elif angle == 2: # 180 deg: (x, y, z) -> (x, -y, -z)
            return (x, -y, -z)
        else: # 270 deg: (x, y, z) -> (x, z, -y)
            return (x, z, -y)
    elif axis == 1: # Rotate around Y
        if angle == 1: # (x, y, z) -> (z, y, -x)
            return (z, y, -x)
        elif angle == 2: # (x, y, z) -> (-x, y, -z)
            return (-x, y, -z)
        else: # (x, y, z) -> (-z, y, x)
            return (-z, y, x)
    elif axis == 2: # Rotate around Z
        if angle == 1: # (x, y, z) -> (-y, x, z)
            return (-y, x, z)
        elif angle == 2: # (x, y, z) -> (-x, -y, z)
            return (-x, -y, z)
        else: # (x, y, z) -> (y, -x, z)
            return (y, -x, z)
    return v

def _calculate_energy(sequence, chain):
    """Calculate total energy (negative contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            # Check 6 neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                q = (x+dx, y+dy, z+dz)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _get_axis_and_bond(chain, i):
    """Get bond vector and axis for pivot at bond i (between i and i+1)."""
    p1 = chain[i]
    p2 = chain[i+1]
    bond = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
    if bond[0] != 0:
        axis = 0
    elif bond[1] != 0:
        axis = 1
    else:
        axis = 2
    return bond, axis

def _propose_pivot(chain, i, rng):
    """
    Propose a pivot move at bond i.
    Returns (new_chain, valid)
    """
    L = len(chain)
    if i < 1 or i > L - 2:
        return None, False
    
    bond, axis = _get_axis_and_bond(chain, i)
    angle = rng.choice([1, 2, 3])
    
    pivot_point = chain[i]
    
    # Generate new tail
    new_tail = []
    for j in range(i+1, L):
        p = chain[j]
        rel = (p[0] - pivot_point[0], p[1] - pivot_point[1], p[2] - pivot_point[2])
        rot_rel = _rotate_vector(rel, axis, angle)
        new_p = (pivot_point[0] + rot_rel[0], 
                 pivot_point[1] + rot_rel[1], 
                 pivot_point[2] + rot_rel[2])
        new_tail.append(new_p)
    
    # Check collisions with head (0..i)
    # Head is chain[0..i]
    # We can create a set of head points
    head_set = set(chain[:i+1])
    
    for np in new_tail:
        if np in head_set:
            return None, False
            
    new_chain = chain[:i+1] + new_tail
    return new_chain, True

def _energy_delta(sequence, chain, new_chain, i):
    """
    Calculate energy delta for a pivot move at bond i.
    Only contacts between head (0..i) and tail (i+1..L-1) change.
    """
    L = len(chain)
    # Build index map for new_chain
    # We only need to check neighbors of the tail in the new chain
    # and neighbors of the tail in the old chain
    
    # To be safe and simple, we can just recalculate energy for the whole chain?
    # L is max 136. 
    # _calculate_energy is O(L * 6) ~ 800 ops.
    # If we do this for every move, and we do 10000 moves, that's 8M ops.
    # Python can do ~10M-50M simple ops per second.
    # So 8M ops is < 1 second.
    # This is acceptable for the time budget.
    
    old_e = _calculate_energy(sequence, chain)
    new_e = _calculate_energy(sequence, new_chain)
    return new_e - old_e

def _anneal(sequence, chain, start_time, deadline, rng):
    """
    Run simulated annealing on the given chain.
    Returns the best chain found.
    """
    L = len(sequence)
    current_chain = list(chain)
    current_energy = _calculate_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Initial temperature
    # Energy range is roughly -L to 0.
    # Let's start with T = 1.0 and decay.
    T = 1.0
    cooling_rate = 0.999
    
    # We will run until deadline
    while time.time() < deadline:
        # Pick a random pivot index
        # Avoid pivoting at the very ends to keep the start fixed? 
        # Actually, fixing the start is good for comparison, but not strictly required.
        # Let's fix index 0. Pivot i can be 1 to L-2.
        i = rng.randint(1, L - 2)
        
        new_chain, valid = _propose_pivot(current_chain, i, rng)
        if not valid:
            T *= cooling_rate
            continue
        
        delta_e = _energy_delta(sequence, current_chain, new_chain, i)
        
        # Metropolis criterion
        if delta_e < 0:
            # Accept
            current_chain = new_chain
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Accept with probability exp(-delta_e / T)
            # delta_e is positive (energy increases, so it's "worse")
            # We want to minimize energy, so we accept if energy decreases (delta < 0).
            # If delta > 0, we accept with prob exp(-delta/T).
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor temperature to allow some exploration? Or go to 0 for pure hill climbing.
            # Let's go to 0 to ensure we find a local minimum at the end.
            # But if we go to 0 too fast, we get stuck.
            # Let's use a linear cooling or geometric.
            # Geometric is fine.

    return best_chain, best_energy

def _greedy_grow(sequence, dimension, rng):
    """
    Generate an initial chain using greedy growth with some randomness.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for axis in range(3):
            for step in (-1, 1):
                q = list(last)
                q[axis] += step
                q = tuple(q)
                if q in occupied:
                    continue
                # Score
                score = rng.random() * 0.5
                if sequence[i] == 'H':
                    # Check neighbors of q for H in occupied
                    x, y, z = q
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        n = (x+dx, y+dy, z+dz)
                        if n in occupied:
                            j = occupied[n]
                            if sequence[j] == 'H':
                                score += 1.0
                # Prefer open space
                free = 0
                x, y, z = q
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    n = (x+dx, y+dy, z+dz)
                    if n not in occupied:
                        free += 1
                score += free * 0.1
                
                options.append((score, q))
        
        if not options:
            return None
            
        options.sort(key=lambda x: x[0], reverse=True)
        # Pick from top 3 to add randomness
        top_k = min(3, len(options))
        chosen = rng.choice(options[:top_k])
        site = chosen[1]
        chain.append(site)
        occupied[site] = i
        
    return chain

def fold(sequence, dimension):
    start_time = time.time()
    # 240s CPU budget for all sequences.
    # We are called once per sequence.
    # Let's assume we have ~10s per sequence to be safe (26 * 10 = 260s).
    # But the problem says "240 s CPU budget for all sequences together".
    # If the evaluator runs them sequentially, we might get more.
    # If parallel, less.
    # Let's aim for ~8-10s per call to be safe.
    deadline = start_time + 9.0
    
    L = len(sequence)
    if L <= 1:
        return [[0, 0, 0]]
    
    rng = random.Random(12345 + L)
    
    best_chain = None
    best_energy = 0 # +inf equivalent for negative energies
    
    # Generate multiple initial candidates
    num_inits = 10
    candidates = []
    for _ in range(num_inits):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
            
    # If greedy failed, try a straight line
    if not candidates:
        candidates.append([(i, 0, 0) for i in range(L)])
        
    # Allocate time for annealing each candidate
    # We have ~9s total.
    # Let's pick the best 3 initial candidates based on energy.
    scored_candidates = []
    for c in candidates:
        e = _calculate_energy(sequence, c)
        scored_candidates.append((e, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 3
    top_candidates = [c for e, c in scored_candidates[:3]]
    
    # Run annealing on each
    # Time per candidate: 9s / 3 = 3s
    time_per_candidate = 3.0
    
    for i, init_chain in enumerate(top_candidates):
        cand_start = time.time()
        cand_deadline = cand_start + time_per_candidate
        
        chain, energy = _anneal(sequence, init_chain, cand_start, cand_deadline, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Check global deadline
        if time.time() > deadline:
            break
            
    if best_chain is None:
        # Return the best initial if annealing failed
        if scored_candidates:
            best_chain = scored_candidates[0][1]
        else:
            best_chain = [(i, 0, 0) for i in range(L)]
            
    return [list(p) for p in best_chain]
