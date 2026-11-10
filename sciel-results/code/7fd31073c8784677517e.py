"""HP lattice protein folding on the square lattice with local search."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Small preference for more free space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
    return chain


def _local_search(sequence, chain, max_steps=5000):
    """Hill-climbing local search with random restarts within the chain."""
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    where = {p: i for i, p in enumerate(best_chain)}
    
    step = 0
    no_improve = 0
    while step < max_steps and no_improve < 200 and time.time() < _DEADLINE:
        step += 1
        n = len(best_chain)
        # Pick a random residue to try moving
        i = random.randint(1, n - 2)  # avoid endpoints
        old_pos = best_chain[i]
        old_occupied = where[old_pos]
        
        # Try all 4 neighbours
        best_new_pos = None
        best_new_energy = best_energy
        
        for new_pos in _neighbours(old_pos):
            if new_pos == old_pos:
                continue
            # Check if new_pos is occupied by another residue
            if new_pos in where and where[new_pos] != i:
                continue
            
            # Tentatively move
            del where[old_pos]
            best_chain[i] = new_pos
            where[new_pos] = i
            
            # Check self-avoidance: make sure no two residues are at same pos
            # (already handled by where dict)
            # Compute energy change efficiently
            # Remove old contacts involving residue i
            old_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            new_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            # Also need to check if moving i breaks or creates contacts between other pairs?
            # No, only contacts involving i change when i moves.
            delta = (new_contacts - old_contacts)
            new_energy = best_energy + delta  # energy = -contacts, so delta in energy = -delta_contacts
            # Wait: energy = -contacts. If contacts increase by k, energy decreases by k.
            # So new_energy = best_energy - (new_contacts - old_contacts)
            new_energy = best_energy - delta
            
            if new_energy < best_new_energy:
                best_new_energy = new_energy
                best_new_pos = new_pos
            
            # Revert
            del where[new_pos]
            best_chain[i] = old_pos
            where[old_pos] = i
        
        if best_new_pos is not None:
            best_chain[i] = best_new_pos
            del where[old_pos]
            where[best_new_pos] = i
            best_energy = best_new_energy
            no_improve = 0
        else:
            no_improve += 1
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, max_steps=10000):
    """Simulated annealing for further refinement."""
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    where = {p: i for i, p in enumerate(best_chain)}
    current_energy = best_energy
    temperature = 1.0
    cooling = 0.999
    
    step = 0
    while step < max_steps and time.time() < _DEADLINE:
        step += 1
        n = len(best_chain)
        i = random.randint(1, n - 2)
        old_pos = best_chain[i]
        
        for new_pos in _neighbours(old_pos):
            if new_pos == old_pos:
                continue
            if new_pos in where and where[new_pos] != i:
                continue
            
            # Tentatively move
            del where[old_pos]
            best_chain[i] = new_pos
            where[new_pos] = i
            
            # Compute new energy
            # Remove old contacts involving i
            old_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            new_contacts = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            delta_contacts = new_contacts - old_contacts
            new_energy = current_energy - delta_contacts
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / max(temperature, 0.01)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                break
            else:
                # Revert
                del where[new_pos]
                best_chain[i] = old_pos
                where[old_pos] = i
                break
        
        temperature *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Multiple greedy growths
    num_greedy = max(50, min(200, int(30 * 1000 / max(n, 1))))
    for _ in range(num_greedy):
        if time.time() >= _DEADLINE:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = 0
        return [list(p) for p in best_chain]
    
    # Phase 2: Local search on the best greedy result
    best_chain, best_energy = _local_search(sequence, best_chain, max_steps=3000)
    
    # Phase 3: Simulated annealing for further refinement
    if time.time() < _DEADLINE - 10:
        best_chain, best_energy = _simulated_annealing(sequence, best_chain, max_steps=5000)
    
    # Phase 4: A few more local search passes from the SA result
    if time.time() < _DEADLINE - 5:
        best_chain, best_energy = _local_search(sequence, best_chain, max_steps=2000)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
