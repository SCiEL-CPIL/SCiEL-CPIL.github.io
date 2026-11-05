"""HP lattice protein folding using Monte Carlo local search."""
# EVOLVE-BLOCK-START
import random
import time
import sys


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


def _valid_move(sequence, chain, occupied, i, new_site):
    # Check self-avoidance
    if new_site in occupied:
        return False
    # Check unit step
    old_site = chain[i]
    if abs(old_site[0] - new_site[0]) + abs(old_site[1] - new_site[1]) != 1:
        return False
    return True


def _greedy_grow(sequence, dimension, rng):
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


def _mc_search(sequence, chain, rng, max_steps=5000):
    """Monte Carlo local search to improve the conformation."""
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    for step in range(max_steps):
        # Pick a random internal residue to move
        i = rng.randint(1, len(chain) - 2)
        old_site = chain[i]
        
        # Try all 4 neighbours
        for new_site in _neighbours(old_site):
            if not _valid_move(sequence, chain, occupied, i, new_site):
                continue
            
            # Compute energy change
            delta = 0
            # Remove old contacts involving residue i
            if sequence[i] == "H":
                for q in _neighbours(old_site):
                    j = occupied.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                        delta += 1  # removing a contact increases energy
            
            # Add new contacts involving residue i
            if sequence[i] == "H":
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j != i - 1 and j != i + 1 and sequence[j] == "H":
                        delta -= 1  # adding a contact decreases energy
            
            # Accept if better or with some probability (simulated annealing-like)
            if delta < 0 or rng.random() < 0.3:
                # Perform the move
                occupied.pop(old_site)
                chain[i] = new_site
                occupied[new_site] = i
                
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation from multiple greedy growths followed by MC refinement."""
    start_time = time.time()
    deadline = start_time + 150  # safety margin before 210s limit
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1  # start with a high (bad) energy
    
    # First, try greedy growths to get good initial conformations
    num_greedy = 50
    for _ in range(num_greedy):
        if time.time() > deadline:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Now apply MC search to the best conformation found
    if time.time() < deadline:
        # Use remaining time for MC search
        remaining_time = deadline - time.time()
        # Estimate steps per second and set max_steps accordingly
        # For a chain of length N, each step involves O(1) work roughly
        max_steps = int(remaining_time * 5000)  # conservative estimate
        max_steps = max(max_steps, 1000)
        max_steps = min(max_steps, 50000)
        
        best_chain, best_energy = _mc_search(sequence, best_chain, rng, max_steps)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
