"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


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
    """One randomized greedy chain growth."""
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


def _sa_improve(sequence, chain, rng, time_budget=1.0):
    """Simulated annealing with incremental energy updates."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.99
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in occupied:
                continue
            
            # Remove old position
            del occupied[old_pos]
            
            # Calculate energy change
            delta = 0
            # Remove contacts at old_pos
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        delta += 1 # Energy increases (less negative) if we remove a contact? 
                        # Energy = -contacts. Removing contact -> energy goes UP (e.g. -5 to -4).
                        # So delta = new - old.
                        # If we lose a contact, new_energy = - (contacts - 1) = -contacts + 1.
                        # old_energy = -contacts.
                        # delta = 1.
            
            # Add new position
            occupied[new_pos] = i
            current_chain[i] = new_pos
            
            # Add contacts at new_pos
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        delta -= 1 # Gaining a contact -> energy goes DOWN (more negative).
            
            if delta <= 0 or rng.random() < (10 ** (-delta / T)):
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Revert
                del occupied[new_pos]
                occupied[old_pos] = i
                current_chain[i] = old_pos
        
        if not moved:
            continue
            
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing with strict time budget."""
    rng = random.Random(12345 + len(sequence))
    
    # Estimate remaining time per sequence
    # Assume 26 sequences total. 300s budget.
    # To be safe, we assume we are at the start and have 300s, or we track global time.
    # Since fold is called per sequence, we don't know the index.
    # We must use a fixed budget per sequence that sums to < 300s.
    # 26 sequences * 10s = 260s. Safe margin.
    seq_budget = 10.0
    
    best, best_energy = None, 0
    
    # 1. Greedy attempts
    t_start = time.time()
    num_greedy = 20
    for _ in range(num_greedy):
        if time.time() - t_start > seq_budget * 0.2:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
    
    # 2. SA improvement
    t_sa_start = time.time()
    sa_time_left = seq_budget - (time.time() - t_start)
    if sa_time_left > 0.1:
        best, best_energy = _sa_improve(sequence, best, rng, time_budget=sa_time_left)
    
    # 3. More greedy + short SA if time remains?
    # Given the tight budget, stick to one good SA run.
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
