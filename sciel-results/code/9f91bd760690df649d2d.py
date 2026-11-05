"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng, weight=1.0, free_weight=0.05):
    """
    Grow a chain greedily, preferring sites that create new H-H contacts.
    Returns a list of tuples or None if stuck.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Calculate gain: new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = weight * gain + free_weight * free + rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search(sequence, dimension, chain, rng, max_iters=50):
    """
    Simple local search: try to swap or shift segments to improve energy.
    For simplicity, we'll do a basic random restart within the search.
    Returns the best chain found.
    """
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(sequence)
    
    for _ in range(max_iters):
        # Try a random segment reversal or shift
        if n < 4:
            break
        i = rng.randint(0, n - 3)
        j = rng.randint(i + 1, n - 1)
        # Reverse segment i..j
        new_chain = list(chain)
        new_chain[i:j+1] = reversed(new_chain[i:j+1])
        
        # Check self-avoidance
        seen = set()
        valid = True
        for k, p in enumerate(new_chain):
            if p in seen:
                valid = False
                break
            seen.add(p)
            if k > 0:
                diff = sum(abs(a-b) for a, b in zip(new_chain[k], new_chain[k-1]))
                if diff != 1:
                    valid = False
                    break
        
        if valid:
            e = _energy(sequence, new_chain)
            if e < best_energy:
                best_energy = e
                best_chain = new_chain
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a 3D lattice conformation.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # For very short chains, just return a straight line
    if n <= 3:
        return [[i, 0, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is the worst (no contacts)
    
    # Count H residues to estimate potential
    h_count = sequence.count("H")
    
    # Multiple random restarts with greedy growth
    rng = random.Random(42)
    
    # Number of restarts depends on chain length and time budget
    # Shorter chains can afford more restarts
    if n <= 50:
        max_restarts = 500
    elif n <= 100:
        max_restarts = 200
    else:
        max_restarts = 100
    
    restarts_done = 0
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Vary the greedy parameters slightly
        weight = 0.8 + rng.random() * 0.4
        free_weight = 0.02 + rng.random() * 0.08
        
        chain = _grow_greedy(sequence, dimension, rng, weight, free_weight)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Local search to improve
        if time.time() < deadline - 10.0:
            chain = _local_search(sequence, dimension, chain, rng, max_iters=30)
            energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        restarts_done += 1
    
    # If no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
