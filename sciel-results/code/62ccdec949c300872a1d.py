"""HP Lattice Protein Folding using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _greedy_chain(sequence, dimension, rng):
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        gain += 1
            options.append((gain + rng.random(), site))
        if not options:
            return None
        options.sort(reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    start_time = time.time()
    # Budget: ~15s per sequence
    deadline = start_time + 15.0
    
    n = len(sequence)
    
    # 1. Initialization
    best_chain = None
    best_energy = 0
    
    for attempt in range(10):
        rng = random.Random(1000 + attempt + n)
        chain = _greedy_chain(sequence, dimension, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        return [[i, 0] for i in range(n)]

    current_chain = best_chain
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.99
    
    # Pre-compute H flags for speed
    is_h = [c == 'H' for c in sequence]
    
    while time.time() < deadline:
        # Kink Move
        i = random.randint(0, n - 2)
        # Limit segment length to ensure speed
        max_len = min(n - i, 15) 
        j = random.randint(i + 1, i + max_len)
        
        rot = random.choice([1, 2, 3])
        
        segment = current_chain[i:j+1]
        pivot = current_chain[i]
        
        # Generate new segment
        new_segment = []
        for p in segment:
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            if rot == 1:
                nx, ny = -dy, dx
            elif rot == 2:
                nx, ny = -dx, -dy
            else:
                nx, ny = dy, -dx
            new_segment.append((pivot[0] + nx, pivot[1] + ny))
        
        # Remove old segment from occupied
        for k in range(i, j+1):
            del occupied[current_chain[k]]
            
        # Check collisions
        valid = True
        for p in new_segment:
            if p in occupied:
                valid = False
                break
                
        if valid:
            # Calculate Delta Energy
            # Contacts between segment and rest
            old_contacts = 0
            for k in range(i, j+1):
                if is_h[k]:
                    for q in _neighbours(current_chain[k]):
                        idx = occupied.get(q)
                        if idx is not None and is_h[idx]:
                            old_contacts += 1
                            
            new_contacts = 0
            for k, p_new in zip(range(i, j+1), new_segment):
                if is_h[k]:
                    for q in _neighbours(p_new):
                        idx = occupied.get(q)
                        if idx is not None and is_h[idx]:
                            new_contacts += 1
                            
            delta_e = -(new_contacts - old_contacts)
            
            # Acceptance
            if delta_e < 0 or random.random() < math.exp(delta_e / T):
                # Apply move
                for k, p_new in zip(range(i, j+1), new_segment):
                    occupied[p_new] = k
                    current_chain[k] = p_new
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                # Revert
                for k in range(i, j+1):
                    occupied[current_chain[k]] = k
        else:
            # Revert
            for k in range(i, j+1):
                occupied[current_chain[k]] = k
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
