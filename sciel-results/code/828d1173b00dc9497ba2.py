"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _local_move(sequence, chain, rng):
    """
    Try a local move: pick a random interior residue and try to move it to an adjacent empty site.
    Returns (new_chain, energy_delta) if valid, else (None, 0).
    """
    n = len(chain)
    if n < 3:
        return None, 0
    
    # Pick a random interior position (not endpoints)
    idx = rng.randint(1, n - 2)
    current = chain[idx]
    
    # Try each neighbor
    for new_pos in _neighbours(current):
        # Check if new_pos is unoccupied (except by the residue itself)
        # We need to check all residues except idx
        valid = True
        for j, p in enumerate(chain):
            if j == idx:
                continue
            if p == new_pos:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check that the move doesn't break connectivity:
        # The residue at idx must still be adjacent to idx-1 and idx+1 after the move
        # Actually, we're moving residue idx to new_pos, so we need:
        # - new_pos adjacent to chain[idx-1]
        # - new_pos adjacent to chain[idx+1]
        if not any(abs(new_pos[a] - chain[idx-1][a]) == 1 for a in range(len(new_pos))):
            continue
        if not any(abs(new_pos[a] - chain[idx+1][a]) == 1 for a in range(len(new_pos))):
            continue
        
        # Compute energy delta
        # Remove contacts involving idx, add new contacts involving idx at new_pos
        where_old = {p: i for i, p in enumerate(chain)}
        
        # Count H-H contacts involving idx in old position
        old_contacts = 0
        if sequence[idx] == 'H':
            for q in _neighbours(current):
                j = where_old.get(q)
                if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
        
        # Build new chain
        new_chain = list(chain)
        new_chain[idx] = new_pos
        where_new = {p: i for i, p in enumerate(new_chain)}
        
        # Count H-H contacts involving idx in new position
        new_contacts = 0
        if sequence[idx] == 'H':
            for q in _neighbours(new_pos):
                j = where_new.get(q)
                if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
        
        energy_delta = -(new_contacts - old_contacts)
        
        return new_chain, energy_delta
    
    return None, 0


def _anneal(sequence, chain, rng, time_limit):
    """
    Simulated annealing on the chain.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_temp = 1.0
    end_temp = 0.01
    cooling = 0.999
    
    temp = start_temp
    steps = 0
    max_steps = 50000
    
    while temp > end_temp and steps < max_steps and _remaining_time() > 1.0:
        # Try a local move
        new_chain, delta = _local_move(sequence, current_chain, rng)
        
        if new_chain is not None:
            new_energy = current_energy + delta
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            if new_energy < current_energy or (delta > 0 and rng.random() < pow(2.718281828, -delta / temp)):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
        
        temp *= cooling
        steps += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    
    # First, do a few greedy growths to get a good starting point
    best_chain = None
    best_energy = 1
    
    # Do initial greedy growths
    num_greedy = 50
    for _ in range(num_greedy):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Now apply simulated annealing to improve
    if _remaining_time() > 2.0:
        improved = _anneal(sequence, best_chain, rng, _remaining_time() - 2.0)
        improved_energy = _energy(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # If we still have time, try more annealing rounds
    if _remaining_time() > 5.0:
        # Try a few more greedy starts and anneal them
        for _ in range(10):
            if _remaining_time() < 2.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
            
            # Anneal this chain
            if _remaining_time() > 3.0:
                improved = _anneal(sequence, best_chain, rng, _remaining_time() - 3.0)
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_chain = improved
                    best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
