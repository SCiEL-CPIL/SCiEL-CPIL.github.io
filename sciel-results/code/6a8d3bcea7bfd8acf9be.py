"""Rewritten HP lattice folding program using simulated annealing with chain growth."""
# EVOLVE-BLOCK-START
import random
import math
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

def _grow_greedy(sequence, dimension, rng, bias=0.3):
    """Grow chain with greedy preference for H-H contacts and free space."""
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
            score = gain + bias * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _simulate_annealing(sequence, dimension, start_chain, max_steps=20000, t_start=1.0, t_end=0.01, cooling=0.999):
    """Simulated annealing on the chain to minimize energy."""
    chain = list(start_chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    n = len(sequence)
    if n < 4:
        return best_chain, best_energy
    
    temperature = t_start
    step = 0
    while temperature > t_end and step < max_steps:
        step += 1
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try moving residue i to a neighbouring free site
        moves = []
        for site in _neighbours(old_pos):
            if site in where:
                # Check if the occupying residue is i itself
                if where[site] == i:
                    continue
                # Check if we can swap or if it's free
                # Actually, we need to remove i first, then check if site is free
                pass
            # We need to be careful: remove i, then check if site is free
        # Better approach: remove i, check neighbours, try to place
        del where[old_pos]
        valid_sites = []
        for site in _neighbours(old_pos):
            if site not in where:
                valid_sites.append(site)
        
        if not valid_sites:
            where[old_pos] = i
            continue
        
        new_pos = valid_sites[rng.randint(0, len(valid_sites) - 1)]
        where[new_pos] = i
        chain[i] = new_pos
        
        new_energy = _energy(sequence, chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            del where[new_pos]
            chain[i] = old_pos
            where[old_pos] = i
        
        temperature *= cooling
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold sequence using multiple growth strategies followed by simulated annealing."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst (unfolded)
    
    # Generate initial conformations using different biases
    biases = [0.1, 0.3, 0.5, 0.7, 1.0]
    initial_chains = []
    for bias in biases:
        for _ in range(2):
            chain = _grow_greedy(sequence, dimension, rng, bias=bias)
            if chain is not None:
                initial_chains.append(chain)
    
    # If no valid chains, use a straight line
    if not initial_chains:
        initial_chains.append([tuple([i] + [0] * (dimension - 1)) for i in range(n)])
    
    # Apply simulated annealing to each initial conformation
    for chain in initial_chains:
        annealed_chain, energy = _simulate_annealing(sequence, dimension, chain, 
                                                      max_steps=5000, t_start=1.0, t_end=0.05, cooling=0.9995)
        if energy < best_energy:
            best_energy = energy
            best_chain = annealed_chain
    
    # Also try a few more growth+annealing cycles if time permits
    start_time = time.time()
    time_budget = 10  # seconds per sequence
    
    while time.time() - start_time < time_budget:
        chain = _grow_greedy(sequence, dimension, rng, bias=rng.uniform(0.1, 1.0))
        if chain is None:
            continue
        annealed_chain, energy = _simulate_annealing(sequence, dimension, chain,
                                                      max_steps=3000, t_start=1.0, t_end=0.05, cooling=0.9995)
        if energy < best_energy:
            best_energy = energy
            best_chain = annealed_chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
