"""HP lattice protein folding on the square lattice."""
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
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 185s as a safe deadline to ensure we finish before kill.
    return 185.0 - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the energy of a conformation: -number of H-H contacts."""
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
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, rng, max_iters=5000):
    """
    Local search: try moving individual residues or flipping segments.
    Use delta energy computation for speed.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    # Precompute contact set for delta energy
    def compute_contacts(where, seq):
        contacts = set()
        for i, p in enumerate(seq):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    contact_set = compute_contacts(where, chain)
    
    def delta_energy_move(i, new_pos, where, chain, seq, contact_set):
        """Compute delta energy if we move residue i to new_pos."""
        old_pos = chain[i]
        if new_pos == old_pos:
            return 0, contact_set
        
        # Check if new_pos is valid (not occupied by another residue)
        if new_pos in where:
            j = where[new_pos]
            if j != i:
                return None, contact_set
        
        # Remove old contacts involving i
        old_contacts = set()
        for (a, b) in contact_set:
            if a == i or b == i:
                old_contacts.add((a, b))
        
        # Add new contacts involving i at new_pos
        new_contacts = set()
        if seq[i] == "H":
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i:
                    if seq[j] == "H":
                        pair = (min(i, j), max(i, j))
                        new_contacts.add(pair)
        
        # Delta energy: -len(new_contacts) + len(old_contacts)
        delta = len(old_contacts) - len(new_contacts)
        
        # Update contact_set
        new_contact_set = contact_set - old_contacts | new_contacts
        return delta, new_contact_set
    
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0
    
    for iteration in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Try to move a random residue
        i = rng.randint(1, n - 2)  # Don't move ends
        old_pos = chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                j = where[new_pos]
                if j != i:
                    continue
            
            delta, new_contact_set = delta_energy_move(i, new_pos, where, chain, sequence, contact_set)
            if delta is None:
                continue
            
            if delta < 0:
                # Accept the move
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
            elif delta == 0 and rng.random() < 0.3:
                # Accept neutral moves sometimes
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                no_improve += 1
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        if no_improve > 200:
            break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, max_iters=10000):
    """
    Simulated annealing to escape local minima.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    def compute_contacts(where, seq, chain):
        contacts = set()
        for i, p in enumerate(chain):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    contact_set = compute_contacts(where, sequence, chain)
    
    def delta_energy_move(i, new_pos, where, chain, seq, contact_set):
        old_pos = chain[i]
        if new_pos == old_pos:
            return 0, contact_set
        
        if new_pos in where:
            j = where[new_pos]
            if j != i:
                return None, contact_set
        
        old_contacts = set()
        for (a, b) in contact_set:
            if a == i or b == i:
                old_contacts.add((a, b))
        
        new_contacts = set()
        if seq[i] == "H":
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i:
                    if seq[j] == "H":
                        pair = (min(i, j), max(i, j))
                        new_contacts.add(pair)
        
        delta = len(old_contacts) - len(new_contacts)
        new_contact_set = contact_set - old_contacts | new_contacts
        return delta, new_contact_set
    
    best_chain = list(chain)
    best_energy = current_energy
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    for iteration in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        if T < T_min:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                j = where[new_pos]
                if j != i:
                    continue
            
            delta, new_contact_set = delta_energy_move(i, new_pos, where, chain, sequence, contact_set)
            if delta is None:
                continue
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                # Accept the move
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                moved = True
                break
        
        if not moved:
            pass  # Stay in place
        
        T *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Randomized greedy growth with local search refinement
    2. Simulated annealing from good starting points
    """
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so start with 0 (no contacts)
    
    # Strategy 1: Multiple greedy growths with local search
    n_growth = 300
    for _ in range(n_growth):
        if _time_remaining() < 10.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Refine with local search
        refined_chain, refined_energy = _local_search(sequence, chain, rng, max_iters=3000)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Strategy 2: Simulated annealing from the best chain found
    if best_chain is not None and _time_remaining() > 15.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, max_iters=15000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Strategy 3: More greedy growths if time allows
    while _time_remaining() > 10.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        refined_chain, refined_energy = _local_search(sequence, chain, rng, max_iters=2000)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
