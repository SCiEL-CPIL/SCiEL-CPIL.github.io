"""HP lattice protein folding using Simulated Annealing with Pivot moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _neighbours(point):
    """Yield 4-neighbors of a 2D point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
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
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _initial_contacts(sequence, chain):
    """Find all H-H contacts in the initial chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

def _apply_pivot(chain, occupied, contacts, i, j, direction):
    """
    Apply pivot move. Returns True if successful.
    Modifies chain, occupied, contacts in-place if successful.
    """
    n = len(chain)
    if i >= j:
        return False
    
    pivot_point = chain[i]
    px, py = pivot_point
    
    # Store old positions for the segment
    old_positions = []
    for k in range(i+1, j+1):
        old_positions.append(chain[k])
        
    # Calculate new positions
    new_positions = []
    for k in range(i+1, j+1):
        ox, oy = old_positions[k - (i+1)]
        rx, ry = ox - px, oy - py
        if direction == 1: # CW
            nx, ny = px + ry, py - rx
        else: # CCW
            nx, ny = px - ry, py + rx
        new_positions.append((nx, ny))
        
    # Check for collisions
    for pos in new_positions:
        if pos in occupied and pos not in old_positions:
            return False
            
    # Update occupied
    for k in range(i+1, j+1):
        del occupied[old_positions[k - (i+1)]]
        
    for k in range(i+1, j+1):
        occupied[new_positions[k - (i+1)]] = k
        chain[k] = new_positions[k - (i+1)]
        
    # Update contacts
    # 1. Remove contacts involving any index in [i+1, j]
    for c in list(contacts):
        u, v = c
        if (i+1 <= u <= j) or (i+1 <= v <= j):
            contacts.remove(c)
            
    # 2. Add new contacts involving any index in [i, j] (including pivot i)
    # Because pivot i's neighbors changed relative to the rest? 
    # No, pivot i is fixed. But the residues in [i+1, j] moved.
    # So contacts between [i+1, j] and the rest of the chain (including i) might change.
    # Also contacts within [i+1, j] might change.
    
    # Check indices in [i, j]
    for idx in range(i, j+1):
        if sequence[idx] != "H":
            continue
        p = chain[idx]
        for q in _neighbours(p):
            neighbor_idx = occupied.get(q)
            if neighbor_idx is None:
                continue
            if abs(idx - neighbor_idx) <= 1:
                continue
            if sequence[neighbor_idx] != "H":
                continue
            u, v = min(idx, neighbor_idx), max(idx, neighbor_idx)
            contacts.add((u, v))
            
    return True

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n < 2:
        return [[0, 0]] * n
        
    rng = random.Random(42 + n)
    
    # 1. Initial Conformation
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
        
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = _initial_contacts(sequence, chain)
    current_energy = -len(contacts)
    
    best_chain = chain[:]
    best_energy = current_energy
    
    # 2. Simulated Annealing
    start_time = time.time()
    max_time = 10.0 # 10s per sequence
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < max_time:
        move_type = rng.random()
        
        if move_type < 0.8:
            # Pivot
            i = rng.randint(0, n - 2)
            max_len = n - 1 - i
            if max_len < 2:
                continue
            length = rng.randint(2, max_len)
            j = i + length
            direction = rng.choice([1, -1])
            
            # Save state for revert
            old_chain = chain[:]
            old_occupied = occupied.copy()
            old_contacts = contacts.copy()
            old_energy = current_energy
            
            success = _apply_pivot(chain, occupied, contacts, i, j, direction)
            
            if success:
                new_energy = -len(contacts)
                delta_e = new_energy - old_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # Revert
                    chain[:] = old_chain
                    occupied.clear()
                    occupied.update(old_occupied)
                    contacts.clear()
                    contacts.update(old_contacts)
                    
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
