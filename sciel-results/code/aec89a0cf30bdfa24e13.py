import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if abs(i - j) > 1 and sequence[j] == 'H':
                        energy -= 1
    return energy

def _beam_search(sequence, dimension, beam_width=50, max_depth=None):
    if max_depth is None:
        max_depth = len(sequence)
    
    # State: (chain, occupied_set, energy)
    # Start
    start = (0,) * dimension
    initial_chain = [start]
    initial_occ = {start}
    initial_energy = 0
    beam = [(initial_chain, initial_occ, initial_energy)]
    
    for step in range(1, max_depth):
        new_beam = []
        for chain, occ, energy in beam:
            last = chain[-1]
            for q in _neighbours(last):
                if q in occ:
                    continue
                # New chain
                new_chain = chain + [q]
                new_occ = occ | {q}
                # Calculate new energy incrementally? 
                # Or just calculate full energy for simplicity in beam search?
                # Full energy is O(N). N is small in early steps.
                # But beam width is 50. 50 * O(N) is fine.
                new_energy = _energy(sequence, new_chain)
                new_beam.append((new_chain, new_occ, new_energy))
        
        if not new_beam:
            break
            
        # Sort by energy (lower is better)
        # We might want to add some diversity, but energy is the main driver
        new_beam.sort(key=lambda x: x[2])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
    return beam

def _pivot_move(chain, i, axis, angle):
    # Rotate chain[i+1:] around chain[i] parallel to axis
    # angle: 1, 2, or 3 (90, 180, 270 degrees)
    pivot = chain[i]
    new_chain = chain[:i+1]
    
    for p in chain[i+1:]:
        # Translate to origin
        x, y, z = p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]
        
        if axis == 0: # x-axis
            if angle == 1: # 90 deg
                nx, ny, nz = x, -z, y
            elif angle == 2: # 180 deg
                nx, ny, nz = x, -y, -z
            else: # 270 deg
                nx, ny, nz = x, z, -y
        elif axis == 1: # y-axis
            if angle == 1:
                nx, ny, nz = z, y, -x
            elif angle == 2:
                nx, ny, nz = -x, y, -z
            else:
                nx, ny, nz = -z, y, x
        else: # z-axis
            if angle == 1:
                nx, ny, nz = y, -x, z
            elif angle == 2:
                nx, ny, nz = -x, -y, z
            else:
                nx, ny, nz = -y, x, z
        
        new_chain.append((nx + pivot[0], ny + pivot[1], nz + pivot[2]))
        
    return new_chain

def _local_search(sequence, chain, time_limit):
    n = len(chain)
    if n <= 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temp = 10.0
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random pivot index
        i = random.randint(1, n - 2)
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])
        
        new_chain = _pivot_move(current_chain, i, axis, angle)
        
        # Check validity
        # We need to check if new_chain is self-avoiding
        # Efficient check:
        # The head [0..i] is unchanged.
        # The tail [i+1..n-1] is rotated.
        # We need to ensure:
        # 1. Tail doesn't intersect itself (it was self-avoiding, rotation preserves distances, so it remains self-avoiding IF the rotation is a rigid body move? No, rotation is rigid, so relative distances are preserved. So tail is self-avoiding.)
        # 2. Tail doesn't intersect Head.
        
        head_set = set(current_chain[:i+1])
        # Check intersection
        valid = True
        for p in new_chain[i+1:]:
            if p in head_set:
                valid = False
                break
        
        if not valid:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < pow(2.71828, -delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temp *= cooling
        
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Time budget allocation
    # Total budget is ~240s for all sequences.
    # We don't know how many sequences are left, but we can assume we are called once per sequence.
    # Let's allocate a fixed time per call, say 8 seconds.
    # 26 sequences * 8s = 208s. Safe.
    time_per_sequence = 8.0
    
    # 1. Beam Search
    # Beam width 50
    beam = _beam_search(sequence, dimension, beam_width=50)
    
    if not beam:
        # Fallback
        return [[i, 0, 0] for i in range(n)]
        
    # Get top candidates from beam
    # Sort by energy
    beam.sort(key=lambda x: x[2])
    
    # Take top 3 candidates for local search
    candidates = [b[0] for b in beam[:3]]
    
    best_chain = None
    best_energy = 0
    
    # 2. Local Search on each candidate
    # Split time among candidates
    time_per_candidate = time_per_sequence / len(candidates)
    
    for candidate in candidates:
        # Run local search
        result = _local_search(sequence, candidate, time_per_candidate)
        e = _energy(sequence, result)
        
        if best_chain is None or e < best_energy:
            best_chain = result
            best_energy = e
            
    if best_chain is None:
        # Should not happen
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
