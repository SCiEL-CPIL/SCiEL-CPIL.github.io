import time
import random

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate new H-H contacts when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    x, y, z = q
    delta = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _beam_search_fold_astar(sequence, n, deadline, beam_width=200):
    """
    Beam search with heuristic: Contacts * 10 + FreeSpace.
    """
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, contacts in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                delta = _calc_delta_energy(sequence, occupied, i, q)
                new_contacts = contacts + delta
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                # Calculate free space for next step
                free = 0
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r not in new_occupied:
                        free += 1
                
                # Heuristic: prioritize contacts heavily, then free space
                new_score = new_contacts * 100 + free
                
                new_beams.append((new_chain, new_occupied, new_contacts, new_score))
        
        if not new_beams:
            break
        
        # Sort by score (descending)
        new_beams.sort(key=lambda x: x[3], reverse=True)
        
        # Keep top beam_width
        # State: (chain, occupied, contacts)
        beams = [(b[0], b[1], b[2]) for b in new_beams[:beam_width]]
    
    best_chain = None
    best_contacts = -1
    for chain, occupied, contacts in beams:
        if len(chain) == n:
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
                
    return best_chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization: try single-residue moves to improve energy.
    Uses incremental energy updates for speed.
    """
    n = len(sequence)
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    def _get_energy():
        energy = 0
        for i in range(n):
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1 and sequence[j] == 'H':
                        energy -= 1
        return energy

    current_energy = _get_energy()
    improved = True
    max_iter = 2000 # Increase iterations since each is cheaper
    
    while improved and time.time() < deadline:
        improved = False
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            if n < 4:
                break
            i = rng.randint(1, n - 2)
            
            old_pos = chain[i]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if (q not in occupied or occupied[q] == i):
                    if i > 0:
                        prev = chain[i-1]
                        if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                            continue
                    if i < n-1:
                        nxt = chain[i+1]
                        if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                            continue
                    if q not in occupied or occupied[q] == i:
                        valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos:
                continue
            
            # Calculate delta energy
            delta_e = 0
            is_h = sequence[i] == 'H'
            
            if is_h:
                # Remove contacts at old_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e += 1 # We are losing this contact, so energy goes up (less negative)
                
                # Add contacts at new_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (new_pos[0] + dx, new_pos[1] + dy, new_pos[2] + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e -= 1 # We are gaining this contact, so energy goes down (more negative)
            
            new_energy = current_energy + delta_e
            
            if new_energy < current_energy:
                # Accept move
                del occupied[old_pos]
                occupied[new_pos] = i
                chain[i] = new_pos
                current_energy = new_energy
                improved = True
            else:
                # Reject move, no state change needed since we didn't apply it
                pass
    return chain, current_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 220.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    beam_configs = [
        (200, 42), (300, 123), (400, 456), (250, 789), (350, 321),
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold_astar(sequence, n, deadline, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        occupied = {}
        for idx, pos in enumerate(chain):
            occupied[pos] = idx
        
        energy = 0
        for i in range(n):
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1:
                        if sequence[j] == 'H':
                            energy -= 1
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        if n > 10 and best_energy <= -n * 0.45:
            break
    
    if best_chain is not None:
        for round_num in range(5):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        # Fallback: simple greedy
        init_chain = [(0, 0, 0)]
        init_occupied = {(0, 0, 0): 0}
        for j in range(1, n):
            last_pos = init_chain[-1]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in init_occupied:
                    valid_mvs.append(q)
            if not valid_mvs:
                break
            init_chain.append(valid_mvs[0])
            init_occupied[valid_mvs[0]] = j
        
        if len(init_chain) == n:
            best_chain = init_chain
        else:
            best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
