"""Improved HP lattice folding with simulated annealing and time management."""
# EVOLVE-BLOCK-START
import random
import time

# Track global start time for deadline management
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

def _compute_contacts(sequence, chain):
    """Return a set of frozenset pairs of indices that form H-H contacts, and the energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add(frozenset((i, j)))
    return contacts, -len(contacts)

def _simulated_annealing(sequence, chain, rng, max_iters=5000, start_temp=1.0, end_temp=0.01):
    """Simulated annealing with single-residue moves."""
    current = list(chain)
    n = len(current)
    where = {p: i for i, p in enumerate(current)}
    
    current_contacts, current_energy = _compute_contacts(sequence, current)
    
    if n < 2:
        return current
    
    # Precompute which residues are H
    is_h = [c == 'H' for c in sequence]
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, max_iters))
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        best_move = None
        best_delta = 0
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            if not _is_adjacent(new_pos, current[i-1]):
                continue
            if not _is_adjacent(new_pos, current[i+1]):
                continue
            
            # Compute energy delta
            delta = 0
            if is_h[i]:
                # Remove old contacts involving i
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i:
                        pair = frozenset((i, j))
                        if pair in current_contacts:
                            delta += 1  # losing a contact increases energy (less negative)
                # Add new contacts involving i
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        pair = frozenset((i, j))
                        if pair not in current_contacts:
                            delta -= 1  # gaining a contact decreases energy (more negative)
            
            if best_move is None or delta < best_delta:
                best_delta = delta
                best_move = new_pos
        
        if best_move is None:
            continue
        
        new_energy = current_energy + best_delta
        
        # Accept if better, or with probability exp(-delta/temp)
        accept = False
        if new_energy < current_energy:
            accept = True
        elif temp > 0:
            import math
            prob = math.exp(-best_delta / temp)
            accept = rng.random() < prob
        
        if accept:
            # Apply the move
            del where[old_pos]
            where[best_move] = i
            current[i] = best_move
            
            # Update contacts
            if is_h[i]:
                # Remove old contacts
                old_contact_pairs = set()
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        old_contact_pairs.add(frozenset((i, j)))
                for pair in old_contact_pairs:
                    if pair in current_contacts:
                        current_contacts.remove(pair)
                
                # Add new contacts
                for q in _neighbours(best_move):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        pair = frozenset((i, j))
                        current_contacts.add(pair)
            
            current_energy = new_energy
    
    return current

def _pivot_search(sequence, chain, rng, max_iters=500):
    """Pivot-based local search: try kink moves."""
    current = list(chain)
    n = len(current)
    current_energy = _energy(sequence, current)
    where = {p: i for i, p in enumerate(current)}
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
        if n < 4:
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
        
        if new_pos != p1:
            # Check if new_pos is occupied (excluding position i)
            if new_pos in where:
                continue
            
            # Check adjacency
            if _is_adjacent(new_pos, p0) and _is_adjacent(new_pos, p2):
                current[i] = new_pos
                del where[p1]
                where[new_pos] = i
                new_energy = _energy(sequence, current)
                if new_energy < current_energy:
                    current_energy = new_energy
                else:
                    # Revert
                    current[i] = p1
                    del where[new_pos]
                    where[p1] = i
    
    return current

def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing with time management."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    # Determine number of restarts based on sequence length and time
    if n <= 20:
        num_restarts = 150
    elif n <= 50:
        num_restarts = 80
    else:
        num_restarts = 40
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply simulated annealing
        sa_iters = max(200, 2000 - n * 10)
        chain = _simulated_annealing(sequence, chain, rng, max_iters=sa_iters,
                                     start_temp=1.0, end_temp=0.01)
        
        # Apply pivot search for fine-tuning
        chain = _pivot_search(sequence, chain, rng, max_iters=max(20, 200 - n))
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
