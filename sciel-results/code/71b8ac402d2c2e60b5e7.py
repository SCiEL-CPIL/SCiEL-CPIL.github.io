"""HP lattice protein folding on the cubic lattice - Genetic Algorithm."""
# EVOLVE-BLOCK-START
import random
import math
import time
import copy

def _rotate_90(point, axis, direction):
    """
    Rotate a point (relative to origin) by 90 degrees around the given axis.
    axis: 0 (x), 1 (y), or 2 (z)
    direction: 1 or -1
    """
    x, y, z = point
    if axis == 0: # x-axis
        if direction == 1:
            return (x, z, -y)
        else:
            return (x, -z, y)
    elif axis == 1: # y-axis
        if direction == 1:
            return (-z, y, x)
        else:
            return (z, y, -x)
    else: # z-axis
        if direction == 1:
            return (y, -x, z)
        else:
            return (-y, x, z)

def _compute_energy(sequence, chain):
    """Compute energy of the chain."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in [(p[0]+1, p[1], p[2]), (p[0]-1, p[1], p[2]), 
                  (p[0], p[1]+1, p[2]), (p[0], p[1]-1, p[2]), 
                  (p[0], p[1], p[2]+1), (p[0], p[1], p[2]-1)]:
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _initial_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    for _ in range(n - 1):
        current = chain[-1]
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if abs(dx) + abs(dy) + abs(dz) != 1:
                        continue
                    new_pos = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if new_pos not in occupied:
                        candidates.append(new_pos)
        if not candidates:
            return None
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain

def _crack_rejoin(sequence, chain, rng):
    """
    Attempt a crack-rejoin move: break the chain at two points and rejoin them.
    This allows for significant topological changes.
    """
    n = len(chain)
    if n < 4:
        return None, False
    
    # Choose two cut points i < j
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)
    
    # Segments: A = chain[:i+1], B = chain[i+1:j+1], C = chain[j+1:]
    # We want to join A-C and leave B as a dangling part? No, we must maintain a single chain.
    # Standard crack-rejoin: Break at i and j.
    # Original: 0..i | i+1..j | j+1..n-1
    # New: 0..i | j+1..n-1 | i+1..j ? No, the connection must be valid.
    # Let's try: Join end of A (chain[i]) to start of C (chain[j+1])?
    # And end of C (chain[n-1]) to start of B (chain[i+1])?
    # This creates a cycle if not careful, or a different path.
    # Path: 0..i -> j+1..n-1 -> i+1..j
    # Check if chain[i] is adjacent to chain[j+1]
    # Check if chain[n-1] is adjacent to chain[i+1]
    
    p1 = chain[i]
    p2 = chain[j+1]
    p3 = chain[n-1]
    p4 = chain[i+1]
    
    # Check adjacency for new bonds
    def is_adj(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2]) == 1
    
    if not is_adj(p1, p2) or not is_adj(p3, p4):
        return None, False
        
    # Construct new chain
    new_chain = chain[:i+1] + chain[j+1:] + chain[i+1:j+1]
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None, False
        
    energy = _compute_energy(sequence, new_chain)
    return new_chain, energy

def _pivot_move(sequence, chain, rng):
    """
    Attempt a pivot move.
    """
    n = len(chain)
    if n < 2:
        return None, False
    
    k = rng.randint(0, n - 2)
    pivot = chain[k]
    
    tail = chain[k+1:]
    
    axis = rng.randint(0, 2)
    direction = rng.choice([-1, 1])
    
    tail_rel = [(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]) for p in tail]
    tail_rot = [_rotate_90(p, axis, direction) for p in tail_rel]
    new_tail = [(p[0] + pivot[0], p[1] + pivot[1], p[2] + pivot[2]) for p in tail_rot]
    
    new_chain = chain[:k+1] + new_tail
    
    if len(set(new_chain)) != n:
        return None, False
        
    energy = _compute_energy(sequence, new_chain)
    return new_chain, energy

def _local_search(sequence, chain, time_budget, rng):
    """
    Perform local search to improve a chain.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 10.0
    t_final = 0.5
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
            
        progress = min(1.0, elapsed / time_budget)
        temp = t0 + (t_final - t0) * progress
        
        # Randomly choose move type
        r = rng.random()
        if r < 0.5:
            new_chain, new_energy = _pivot_move(sequence, chain, rng)
            if new_chain is None: continue
        elif r < 0.8:
            new_chain, new_energy = _crack_rejoin(sequence, chain, rng)
            if new_chain is None: continue
        else:
            # Simple move: move last residue
            if n < 2: continue
            i = n - 1
            current = chain[i]
            prev = chain[i-1]
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if abs(dx) + abs(dy) + abs(dz) != 1:
                            continue
                        new_pos = (prev[0] + dx, prev[1] + dy, prev[2] + dz)
                        if new_pos == current:
                            continue
                        if new_pos in set(chain[:-1]):
                            continue
                        candidates.append(new_pos)
            if not candidates:
                continue
            new_pos = rng.choice(candidates)
            new_chain = chain[:-1] + [new_pos]
            new_energy = _compute_energy(sequence, new_chain)
            
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return best_chain, best_energy

def _crossover(seq, parent1, parent2, rng):
    """
    Crossover: Take a prefix from parent1 and suffix from parent2, or vice versa.
    We need to ensure the connection is valid.
    """
    n = len(parent1)
    # Choose crossover point
    k = rng.randint(1, n - 2)
    
    # Try Parent1[0:k+1] + Parent2[k+1:n]
    # Check if parent1[k] is adjacent to parent2[k+1]
    def is_adj(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2]) == 1
        
    if is_adj(parent1[k], parent2[k+1]):
        child = parent1[:k+1] + parent2[k+1:]
        if len(set(child)) == n:
            return child
            
    # Try Parent2[0:k+1] + Parent1[k+1:n]
    if is_adj(parent2[k], parent1[k+1]):
        child = parent2[:k+1] + parent1[k+1:]
        if len(set(child)) == n:
            return child
            
    return None

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using Genetic Algorithm.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(54321 + len(sequence))
    
    # Time allocation
    total_time = 15.0
    start_global = time.time()
    
    # Initialize Population
    pop_size = 20
    population = []
    for _ in range(pop_size):
        chain = None
        attempts = 0
        while chain is None and attempts < 50:
            chain = _initial_saw(n, rng)
            attempts += 1
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
        population.append((chain, energy))
        
    # Sort by energy
    population.sort(key=lambda x: x[1])
    
    best_chain, best_energy = population[0]
    
    # Genetic Algorithm Loop
    generations = 0
    while True:
        elapsed = time.time() - start_global
        if elapsed > total_time:
            break
            
        generations += 1
        
        # Select Parents (Tournament Selection)
        def tournament_select(pop, k=3):
            candidates = [pop[rng.randint(0, len(pop)-1)] for _ in range(k)]
            return min(candidates, key=lambda x: x[1])[0]
            
        new_population = []
        
        # Elitism: Keep top few
        for i in range(3):
            new_population.append(population[i])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            
            child = _crossover(sequence, p1, p2, rng)
            
            if child is None:
                # If crossover fails, use a mutated copy of a parent
                parent = p1 if rng.random() < 0.5 else p2
                # Apply a few mutations
                for _ in range(5):
                    new_chain, new_energy = _pivot_move(sequence, parent, rng)
                    if new_chain is not None:
                        parent = new_chain
                child = parent
                # Verify self-avoidance just in case (pivot should guarantee it if valid)
                if len(set(child)) != n:
                    child = list(p1)
            
            # Local Search on Child
            # Allocate small time budget for local search per child
            ls_time = min(0.2, (total_time - (time.time() - start_global)) / 10.0)
            if ls_time > 0.01:
                child, energy = _local_search(sequence, child, ls_time, rng)
            else:
                energy = _compute_energy(sequence, child)
                
            new_population.append((child, energy))
            
        # Update Population
        population = new_population
        population.sort(key=lambda x: x[1])
        
        if population[0][1] < best_energy:
            best_energy = population[0][1]
            best_chain = list(population[0][0])
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
