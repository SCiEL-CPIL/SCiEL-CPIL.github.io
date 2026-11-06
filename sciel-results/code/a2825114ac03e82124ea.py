"""HP lattice folding using simulated annealing with kink moves and time management."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Leave 15s safety margin before 210s wall-clock limit
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

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

def _compute_contacts(sequence, chain):
    """Compute contact counts for incremental energy updates."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contact_count = [0] * n
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contact_count[i] += 1
                    contact_count[j] += 1
    return where, contact_count

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

def _simulated_annealing(sequence, chain, rng, max_iters=5000):
    """Simulated annealing with kink moves and incremental energy updates."""
    current = list(chain)
    n = len(current)
    
    where, contact_count = _compute_contacts(sequence, current)
    current_energy = _energy(sequence, current)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    best_energy = current_energy
    best_chain = list(current)
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Pick a random internal position for kink move
        i = rng.randint(1, n - 2)
        p0 = current[i-1]
        p1 = current[i]
        p2 = current[i+1]
        
        # The other kink position
        dx0 = p0[0] - p1[0]
        dy0 = p0[1] - p1[1]
        dx2 = p2[0] - p1[0]
        dy2 = p2[1] - p1[1]
        new_pos = (p1[0] + dx0 + dx2, p1[1] + dy0 + dy2)
        
        if new_pos == p1:
            T *= cooling_rate
            continue
        
        # Check if new_pos is occupied (excluding position i)
        if new_pos in where and where[new_pos] != i:
            T *= cooling_rate
            continue
        
        # Check adjacency to chain neighbors
        if not _is_adjacent(new_pos, p0) or not _is_adjacent(new_pos, p2):
            T *= cooling_rate
            continue
        
        # Compute energy delta
        delta = 0
        if sequence[i] == "H":
            # Remove old contacts
            for q in _neighbours(p1):
                j = where.get(q)
                if j is not None and j != i:
                    if (j > i + 1 or j < i - 1) and sequence[j] == "H":
                        delta += 1
            # Add new contacts
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i:
                    if (j > i + 1 or j < i - 1) and sequence[j] == "H":
                        delta -= 1
        
        new_energy = current_energy - delta
        
        # Accept or reject
        if new_energy < current_energy or rng.random() < pow(2.718281828, (new_energy - current_energy) / T):
            # Accept the move
            del where[p1]
            where[new_pos] = i
            current[i] = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current)
        
        T *= cooling_rate
    
    return best_chain

def _local_search(sequence, chain, rng, max_iters=2000):
    """Local search with incremental energy updates."""
    current = list(chain)
    n = len(current)
    
    where, contact_count = _compute_contacts(sequence, current)
    current_energy = _energy(sequence, current)
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        # Try all 4 neighbors
        for new_pos in _neighbours(old_pos):
            if new_pos in where and where[new_pos] != i:
                continue
            # Check adjacency to chain neighbors
            if not _is_adjacent(new_pos, current[i-1]):
                continue
            if not _is_adjacent(new_pos, current[i+1]):
                continue
            
            # Compute energy delta
            delta = 0
            if sequence[i] == "H":
                # Remove old contacts
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i:
                        if (j > i + 1 or j < i - 1) and sequence[j] == "H":
                            delta += 1
                # Add new contacts
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i:
                        if (j > i + 1 or j < i - 1) and sequence[j] == "H":
                            delta -= 1
            
            new_energy = current_energy - delta
            if new_energy < current_energy:
                # Accept the move
                del where[old_pos]
                where[new_pos] = i
                current[i] = new_pos
                current_energy = new_energy
                break
    
    return current

def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing + local search with time management."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    # Determine number of restarts based on sequence length and time
    if n <= 20:
        num_restarts = 300
    elif n <= 50:
        num_restarts = 150
    else:
        num_restarts = 75
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply simulated annealing with kink moves
        chain = _simulated_annealing(sequence, chain, rng, max_iters=max(200, 2000 - n * 10))
        
        # Apply local search for final refinement
        chain = _local_search(sequence, chain, rng, max_iters=max(100, 1000 - n * 5))
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
