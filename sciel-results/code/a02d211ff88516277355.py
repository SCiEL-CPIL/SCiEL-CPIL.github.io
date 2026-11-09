"""
HP lattice protein folding on the cubic lattice.
Strategy: Genetic Algorithm with specialized crossover for self-avoiding walks.
Instead of local search (pivots/slides), we maintain a population of chains
and recombine them. This allows escaping local optima by mixing structural
features from different parts of the search space.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

def _neighbours_list(point):
    """Return list of 6 face-neighbours of point in 3D."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, idx, new_pos, occupied):
    """
    Calculate energy change if residue at idx is moved to new_pos.
    Returns delta or None if move invalid.
    """
    n = len(chain)
    if new_pos in occupied:
        return None
    
    old_pos = chain[idx]
    
    # Check connectivity to neighbors
    if idx > 0:
        prev_pos = chain[idx-1]
        if abs(prev_pos[0]-new_pos[0]) + abs(prev_pos[1]-new_pos[1]) + abs(prev_pos[2]-new_pos[2]) != 1:
            return None
    if idx < n - 1:
        next_pos = chain[idx+1]
        if abs(next_pos[0]-new_pos[0]) + abs(next_pos[1]-new_pos[1]) + abs(next_pos[2]-new_pos[2]) != 1:
            return None
            
    # Calculate energy change
    # Only residues near old_pos and new_pos can affect energy
    # Specifically, H residues that are neighbors to old_pos or new_pos
    
    delta = 0
    
    # Residues that might lose contact with old_pos
    # These are H residues at indices j such that |j - idx| > 1 and old_pos is neighbor to chain[j]
    # And residues that might gain contact with new_pos
    
    # To be precise and fast, we check all H residues? No, only those in the neighborhood.
    # Let's identify the set of indices that are neighbors to old_pos or new_pos.
    
    affected_indices = set()
    
    # Find indices j where chain[j] is neighbor to old_pos
    # We don't have a reverse map from pos to idx in occupied? 
    # occupied is {pos: idx}.
    # So we can iterate neighbors of old_pos and new_pos.
    
    for p in _neighbours_list(old_pos):
        if p in occupied:
            j = occupied[p]
            if abs(j - idx) > 1:
                affected_indices.add(j)
                
    for p in _neighbours_list(new_pos):
        if p in occupied:
            j = occupied[p]
            if abs(j - idx) > 1:
                affected_indices.add(j)
                
    # Also, the residue at idx itself might be H and its contacts change.
    # The logic above handles contacts with OTHER residues.
    # If sequence[idx] is H, we need to check if it had a contact with some j, and if it will have one.
    # The set 'affected_indices' contains j's that are spatially close.
    # We need to check pairs (idx, j) for j in affected_indices.
    
    for j in affected_indices:
        if sequence[j] == 'H' and sequence[idx] == 'H':
            # Check if (idx, j) was a contact
            was_contact = False
            # j is in affected_indices, so chain[j] was neighbor to old_pos OR new_pos?
            # Wait, affected_indices contains j if chain[j] is neighbor to old_pos OR new_pos.
            # We need to know specifically if chain[j] was neighbor to old_pos (before) 
            # and if chain[j] is neighbor to new_pos (after).
            
            # Let's re-evaluate.
            # Contact exists if Manhattan distance is 1 and |i-j| > 1.
            
            dist_old = abs(chain[j][0]-old_pos[0]) + abs(chain[j][1]-old_pos[1]) + abs(chain[j][2]-old_pos[2])
            dist_new = abs(chain[j][0]-new_pos[0]) + abs(chain[j][1]-new_pos[1]) + abs(chain[j][2]-new_pos[2])
            
            if dist_old == 1:
                delta += 1  # Lost a contact (energy increases by 1, i.e., -1 becomes 0, so delta is +1)
            if dist_new == 1:
                delta -= 1  # Gained a contact (energy decreases by 1)
                
    return delta

def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            site_neighbors = _neighbours_list(site)
            h_score = 0
            free_score = 0
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _ga_fold(sequence, dimension, time_budget, rng):
    """
    Genetic Algorithm for folding.
    """
    n = len(sequence)
    deadline = time.time() + time_budget
    
    # Initialize population
    pop_size = 10
    population = []
    energies = []
    
    for _ in range(pop_size):
        chain = _greedy_grow(sequence, dimension, rng, bias=1.0 + rng.random())
        if chain:
            e = _energy(sequence, chain)
            population.append(chain)
            energies.append(e)
    
    if not population:
        return [(i, 0, 0) for i in range(n)], 0
    
    # Sort initial population
    sorted_indices = sorted(range(len(population)), key=lambda i: energies[i])
    population = [population[i] for i in sorted_indices]
    energies = [energies[i] for i in sorted_indices]
    
    best_chain = population[0]
    best_energy = energies[0]
    
    generation = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Create new population
        new_population = []
        new_energies = []
        
        # Elitism: keep top 2
        for i in range(min(2, len(population))):
            new_population.append(population[i])
            new_energies.append(energies[i])
        
        # Fill rest with children
        while len(new_population) < pop_size and time.time() < deadline:
            # Selection: Tournament selection
            def select():
                a = rng.randint(0, len(population)-1)
                b = rng.randint(0, len(population)-1)
                if energies[a] <= energies[b]:
                    return population[a]
                else:
                    return population[b]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: Uniform crossover with repair
            # This is tricky for SAWs. 
            # Alternative: Use a "segment swap" or "recombination" that preserves validity.
            # A simpler and robust approach for SAWs in GA is to take a prefix from one and suffix from another,
            # but they must connect.
            
            # Let's try a different crossover: 
            # 1. Pick a random split point k in parent1.
            # 2. Take prefix of parent1 up to k.
            # 3. Regrow the rest using a greedy method biased by the sequence, 
            #    but start from the end of the prefix.
            # This is essentially "regrowth" guided by the GA population's quality.
            
            k = rng.randint(1, n - 2)
            prefix = parent1[:k]
            
            # Regrow suffix
            # We use a greedy grow but starting from the last point of prefix.
            # To make it "genetic", we can bias the greedy grow towards the structure of parent2?
            # Hard to do directly. 
            # Instead, let's just do a high-quality greedy regrowth.
            # This turns GA into a "memetic" algorithm with regrowth.
            
            # To add more diversity, we can sometimes mutate the parent instead of regrowing.
            if rng.random() < 0.3:
                # Mutation: Pivot or Slide
                child = _mutate(sequence, parent1, rng)
            else:
                # Regrowth
                # We need to ensure the regrowth is valid.
                # _greedy_grow starts from (0,0,0). We need a version that starts from a given chain.
                child = _regrow_from_prefix(sequence, prefix, rng)
            
            if child:
                e = _energy(sequence, child)
                new_population.append(child)
                new_energies.append(e)
        
        if not new_population:
            break
            
        # Sort new population
        sorted_indices = sorted(range(len(new_population)), key=lambda i: new_energies[i])
        population = [new_population[i] for i in sorted_indices]
        energies = [new_energies[i] for i in sorted_indices]
        
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_chain = population[0]
            
    return best_chain, best_energy

def _mutate(sequence, chain, rng):
    """
    Perform a local mutation (pivot or slide) on the chain.
    """
    n = len(chain)
    if n < 3:
        return None
    
    move_type = rng.random()
    
    if move_type < 0.5:
        return _pivot_move_fast(sequence, chain, rng)
    else:
        return _slide_move_fast(sequence, chain, rng)

def _pivot_move_fast(sequence, chain, rng):
    """
    Fast pivot move using delta energy.
    """
    n = len(chain)
    if n < 3:
        return None
    
    split = rng.randint(1, n - 2)
    axis = rng.randint(0, 2)
    direction = rng.choice([1, -1])
    
    def rotate(point, axis, direction):
        x, y, z = point
        if axis == 0:
            if direction == 1:
                return (x, -z, y)
            else:
                return (x, z, -y)
        elif axis == 1:
            if direction == 1:
                return (z, y, -x)
            else:
                return (-z, y, x)
        else:
            if direction == 1:
                return (y, -x, z)
            else:
                return (-y, x, z)
    
    prefix = chain[:split]
    suffix = chain[split:]
    pivot = chain[split-1]
    
    new_suffix = []
    for p in suffix:
        tx = p[0] - pivot[0]
        ty = p[1] - pivot[1]
        tz = p[2] - pivot[2]
        rx, ry, rz = rotate((tx, ty, tz), axis, direction)
        new_suffix.append((rx + pivot[0], ry + pivot[1], rz + pivot[2]))
    
    new_chain = prefix + new_suffix
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check unit steps (should be preserved by rotation, but verify)
    for i in range(n - 1):
        p1 = new_chain[i]
        p2 = new_chain[i+1]
        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        if dist != 1:
            return None
            
    return new_chain

def _slide_move_fast(sequence, chain, rng):
    """
    Fast slide move using delta energy.
    """
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    pos = chain[idx]
    neighbors = _neighbours_list(pos)
    rng.shuffle(neighbors)
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for new_pos in neighbors:
        if new_pos in occupied:
            continue
        
        prev_pos = chain[idx - 1]
        next_pos = chain[idx + 1]
        
        d1 = abs(prev_pos[0]-new_pos[0]) + abs(prev_pos[1]-new_pos[1]) + abs(prev_pos[2]-new_pos[2])
        d2 = abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2])
        
        if d1 != 1 or d2 != 1:
            continue
        
        new_chain = chain[:]
        new_chain[idx] = new_pos
        
        if len(set(new_chain)) != n:
            continue
            
        return new_chain
    
    return None

def _regrow_from_prefix(sequence, prefix, rng):
    """
    Regrow the chain from the given prefix.
    """
    n = len(sequence)
    if not prefix:
        return None
        
    chain = prefix[:]
    occupied = {p: i for i, p in enumerate(chain)}
    
    for i in range(len(prefix), n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            site_neighbors = _neighbours_list(site)
            h_score = 0
            free_score = 0
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * 2.0 + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def fold(sequence, dimension):
    """
    Best conformation using Genetic Algorithm with Regrowth.
    """
    n = len(sequence)
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time per sequence
    # We have 26 sequences. Total budget 325s.
    # Average ~12s per sequence.
    # But some are shorter. Let's allocate based on length roughly.
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
        
    if budget < 1.0:
        budget = 0.5
        
    rng = random.Random(12345 + len(sequence))
    
    # Run GA
    best_chain, best_energy = _ga_fold(sequence, dimension, budget, rng)
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
