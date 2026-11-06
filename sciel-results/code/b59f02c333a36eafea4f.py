"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


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


def _local_refine(sequence, chain, dimension, rng, max_steps=50):
    """Very fast local search: try single-residue moves to improve energy."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        
        n = len(chain)
        if n < 3:
            break
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        
        occupied = set(chain)
        occupied.discard(chain[i])
        
        moved = False
        for site in _neighbours(chain[i]):
            if site in occupied:
                continue
            
            new_chain = chain[:]
            new_chain[i] = site
            
            # Check self-avoidance and connectivity
            if len(set(new_chain)) != len(new_chain):
                continue
            
            valid = True
            for j in range(len(new_chain) - 1):
                diff = sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension))
                if diff != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                chain = new_chain
                moved = True
                break
        
        if not moved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with light local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Scale number of attempts based on chain length
    # Longer chains are slower, so use fewer attempts
    if n <= 48:
        num_attempts = 500
    elif n <= 80:
        num_attempts = 300
    else:
        num_attempts = 150
    
    for attempt in range(num_attempts):
        if time.time() > _DEADLINE:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        
        # Only refine if this is a good candidate
        if value < best_energy:
            best_chain = chain
            best_energy = value
        
        # Do light refinement on promising candidates
        if value <= best_energy + 2:  # within 2 contacts of best
            refined = _local_refine(sequence, chain, dimension, rng, max_steps=20)
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
