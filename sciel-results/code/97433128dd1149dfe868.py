import random
import time
import math

# Global time tracking for safety
_START_TIME = time.time()
_GLOBAL_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

def _neighbours(point):
    """Yield 6 neighbors of a 3D lattice point."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _calculate_energy(sequence, chain):
    """Calculate total energy (negative contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

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

def _propose_pivot(chain, i, rng):
    """
    Propose a pivot move at bond i (rotating tail i+1..L-1 around bond i-1..i).
    Returns (new_chain, valid)
    """
    L = len(chain)
    if i < 1 or i > L - 2:
        return None, False
    
    p0 = chain[i-1]
    p1 = chain[i]
    
    # Determine axis of the bond
    bond = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
    if bond[0] != 0:
        axis = 0
    elif bond[1] != 0:
        axis = 1
    else:
        axis = 2
        
    angle = rng.choice([1, 2, 3])
    
    # The pivot point is p1. We rotate the vector from p1 to each tail point around the bond axis.
    # Actually, standard pivot moves rotate the tail around the bond axis passing through p0 and p1.
    # The vector to rotate is relative to the pivot point p1? No, relative to the axis line.
    # Easier: Translate so p1 is origin? No, the axis passes through p0 and p1.
    # Let's use the rotation relative to the vector from p1 to the point, but the axis direction is fixed.
    # Wait, if we rotate around the axis defined by p0-p1, the point p1 stays fixed.
    # So we can just rotate the vector (p - p1) around the bond direction.
    
    pivot_point = p1
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
    head_set = set(chain[:i+1])
    for np in new_tail:
        if np in head_set:
            return None, False
            
    new_chain = chain[:i+1] + new_tail
    return new_chain, True

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
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score
            score = rng.random() * 0.5
            if sequence[i] == 'H':
                # Check neighbors of site for H in occupied
                for n in _neighbours(site):
                    if n in occupied:
                        j = occupied[n]
                        if sequence[j] == 'H':
                            score += 1.0
            # Prefer open space
            free = 0
            for n in _neighbours(site):
                if n not in occupied:
                    free += 1
            score += free * 0.1
            options.append((score, site))
        
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

def _parallel_tempering(sequence, init_chains, deadline, rng):
    """
    Run parallel tempering (replica exchange) on the given chains.
    """
    num_reps = len(init_chains)
    if num_reps == 0:
        return None, 0
    
    # Temperatures: from high to low
    # T_high allows exploration, T_low allows exploitation
    T_high = 2.0
    T_low = 0.1
    # Geometric spacing
    temps = []
    if num_reps > 1:
        ratio = (T_high / T_low) ** (1.0 / (num_reps - 1))
        for i in range(num_reps):
            temps.append(T_low * (ratio ** i))
    else:
        temps = [1.0]
        
    # Sort chains by energy initially?
    # Assign best chain to lowest temp?
    # Or random assignment?
    # Let's sort initial chains by energy and assign best to lowest T (colder) to refine.
    # Actually, PT usually keeps replicas independent but swaps based on energy.
    
    replicas = []
    for i, chain in enumerate(init_chains):
        e = _calculate_energy(sequence, chain)
        replicas.append({
            'chain': chain,
            'energy': e,
            'temp': temps[i]
        })
    
    best_chain = replicas[0]['chain']
    best_energy = replicas[0]['energy']
    
    L = len(sequence)
    
    # Pre-allocate list of pivot indices to pick from?
    # Just use random
    
    while time.time() < deadline:
        # 1. Update each replica
        for r in replicas:
            chain = r['chain']
            e = r['energy']
            T = r['temp']
            
            # Propose pivot
            i = rng.randint(1, L - 2)
            new_chain, valid = _propose_pivot(chain, i, rng)
            
            if valid:
                new_e = _calculate_energy(sequence, new_chain)
                delta = new_e - e
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    r['chain'] = new_chain
                    r['energy'] = new_e
                    if new_e < best_energy:
                        best_energy = new_e
                        best_chain = new_chain
        
        # 2. Attempt swaps between adjacent replicas
        # Swaps are more likely if the hotter replica has lower energy (unlikely) 
        # or if the colder replica has higher energy (likely? no).
        # Acceptance prob: exp(-T_c * (E_h - E_c) - T_h * (E_c - E_h))
        # = exp(-(T_c - T_h) * (E_h - E_c))
        # Let i be colder (lower T), j be hotter (higher T).
        # If E_i > E_j (colder is worse), swap is likely.
        # If E_i < E_j (colder is better), swap is unlikely.
        
        for k in range(num_reps - 1):
            r1 = replicas[k]
            r2 = replicas[k+1]
            
            # Ensure r1 is colder (lower T)
            if r1['temp'] > r2['temp']:
                r1, r2 = r2, r1
                # Note: indices in list don't change, just local vars for logic
                # But we need to swap the contents if accepted.
                # Since we swapped refs, we must be careful.
                # Let's just use the actual objects from the list.
                pass
            
            # Let's stick to list indices to avoid confusion
            rep_a = replicas[k]
            rep_b = replicas[k+1]
            
            # Determine cold and hot
            if rep_a['temp'] < rep_b['temp']:
                cold, hot = rep_a, rep_b
            else:
                cold, hot = rep_b, rep_a
                
            E_c = cold['energy']
            E_h = hot['energy']
            T_c = cold['temp']
            T_h = hot['temp']
            
            # Delta E for swap:
            # Cold gets E_h, Hot gets E_c
            # Delta = T_c * E_h + T_h * E_c - (T_c * E_c + T_h * E_h)
            #       = (T_c - T_h) * (E_h - E_c)
            
            delta_swap = (T_c - T_h) * (E_h - E_c)
            
            if delta_swap < 0 or rng.random() < math.exp(-delta_swap):
                # Swap chains and energies
                cold['chain'], hot['chain'] = hot['chain'], cold['chain']
                cold['energy'], hot['energy'] = hot['energy'], cold['energy']

    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    # 240s CPU budget for all sequences.
    # We are called once per sequence.
    # Safety deadline: 320s global, but per-call we should be faster.
    # If we assume 26 sequences, 320/26 ~ 12s per sequence is tight if sequential.
    # But the prompt says "240 s CPU budget for all sequences together".
    # If the evaluator runs them in a loop, we have ~9s per sequence.
    # If parallel, we have more, but CPU time is the limit.
    # Let's target 8.5s per call to be safe.
    
    call_deadline = start_time + 8.5
    if call_deadline > _GLOBAL_DEADLINE:
        call_deadline = _GLOBAL_DEADLINE
        
    L = len(sequence)
    if L <= 1:
        return [[0, 0, 0]]
    
    rng = random.Random(12345 + L)
    
    # Generate multiple initial candidates
    # 10 initial chains is a good balance.
    num_inits = 10
    candidates = []
    for _ in range(num_inits):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
            
    if not candidates:
        candidates.append([(i, 0, 0) for i in range(L)])
        
    # Run Parallel Tempering on these candidates
    best_chain, best_energy = _parallel_tempering(sequence, candidates, call_deadline, rng)
    
    if best_chain is None:
        # Fallback
        best_chain = candidates[0]
        
    return [list(p) for p in best_chain]
