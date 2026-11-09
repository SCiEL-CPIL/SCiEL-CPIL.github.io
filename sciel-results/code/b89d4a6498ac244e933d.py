"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


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


def _initial_grow(sequence, dimension, rng):
    """Greedy growth to get a valid self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
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
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _delta_energy(sequence, chain, i, new_site):
    """Compute the change in energy if we move residue i to new_site."""
    where = {p: idx for idx, p in enumerate(chain)}
    old_site = chain[i]
    
    # Remove old contacts involving residue i
    delta = 0
    if sequence[i] == "H":
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                delta += 1  # losing a contact
    
    # Add new contacts involving residue i at new_site
    if sequence[i] == "H":
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                delta -= 1  # gaining a contact
    
    return delta


def _anneal(sequence, dimension, rng, max_steps=5000):
    """Simulated annealing starting from a grown chain."""
    chain = _initial_grow(sequence, dimension, rng)
    if chain is None:
        return None
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    t_start = 2.0
    t_end = 0.01
    cooling_rate = (t_end / t_start) ** (1.0 / max_steps)
    temperature = t_start
    
    for step in range(max_steps):
        temperature *= cooling_rate
        
        # Pick a random residue to move
        i = rng.randint(1, len(sequence) - 2)  # avoid endpoints for simplicity
        old_site = chain[i]
        
        # Find a valid new site
        candidates = []
        for site in _neighbours(old_site):
            if site in occupied:
                if site == old_site:
                    continue
                # Check if it's another residue
                other_idx = occupied[site]
                if abs(other_idx - i) == 1:
                    continue  # can't swap with adjacent
                if other_idx == i:
                    continue
            # Check if site is free or is the old site of an adjacent residue (swap)
            if site not in occupied:
                candidates.append(site)
            else:
                other_idx = occupied[site]
                if abs(other_idx - i) == 1:
                    # Try swap
                    other_site = chain[other_idx]
                    if other_site not in occupied or occupied.get(other_site) == other_idx:
                        # Check if other_site is free or belongs to i
                        if other_site not in occupied or occupied[other_site] == i:
                            candidates.append((-1, site, other_idx))  # swap marker
        
        if not candidates:
            continue
        
        choice = rng.choice(candidates)
        
        if isinstance(choice, tuple):
            # Swap move
            _, new_site, other_idx = choice
            other_site = chain[other_idx]
            
            # Validate swap
            if new_site in occupied and occupied[new_site] != i:
                continue
            if other_site in occupied and occupied[other_site] != other_idx:
                continue
            
            # Compute delta energy for swap
            delta = 0
            # Remove old contacts for i and other_idx
            for idx in (i, other_idx):
                if sequence[idx] == "H":
                    for q in _neighbours(chain[idx]):
                        j = occupied.get(q)
                        if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == "H":
                            delta += 1
            # Add new contacts
            new_positions = {i: new_site, other_idx: other_site}
            for idx in (i, other_idx):
                if sequence[idx] == "H":
                    for q in _neighbours(new_positions[idx]):
                        j = occupied.get(q)
                        if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == "H":
                            # But we need to check if j is one of the swapped residues
                            if j in (i, other_idx):
                                continue
                            delta -= 1
            
            # Perform swap
            occupied[new_site] = i
            occupied[other_site] = other_idx
            del occupied[old_site]
            del occupied[other_site]  # already handled
            # Actually, let's just update properly
            chain[i], chain[other_idx] = new_site, other_site
            occupied[new_site] = i
            occupied[other_site] = other_idx
            if old_site != new_site and old_site != other_site:
                del occupied[old_site]
            
            new_energy = current_energy + delta
        else:
            # Single move
            new_site = choice
            if new_site in occupied:
                continue
            
            delta = _delta_energy(sequence, chain, i, new_site)
            new_energy = current_energy + delta
            
            # Perform move
            occupied[new_site] = i
            del occupied[old_site]
            chain[i] = new_site
        
        # Accept or reject
        if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / temperature):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            if isinstance(choice, tuple):
                chain[i], chain[other_idx] = old_site, other_site
                occupied[old_site] = i
                occupied[other_site] = other_idx
                if new_site != old_site and new_site != other_site:
                    del occupied[new_site]
            else:
                chain[i] = old_site
                occupied[old_site] = i
                del occupied[new_site]
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple annealing runs."""
    rng = random.Random(12345 + len(sequence))
    start_time = time.time()
    deadline = start_time + 300  # 300s budget
    
    best_chain = None
    best_energy = 0
    
    # Try multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:  # safety margin
            break
        
        chain = _anneal(sequence, dimension, rng, max_steps=3000)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
