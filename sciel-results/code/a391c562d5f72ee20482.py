"""HP lattice protein folding on the cubic lattice - Genetic Algorithm with Local Search."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _compute_energy(sequence, chain):
    """Compute energy of the chain."""
    n = len(chain)
    if n == 0:
        return 0
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

def _is_valid_chain(chain):
    """Check if chain is self-avoiding."""
    if len(chain) != len(set(chain)):
        return False
    for i in range(len(chain) - 1):
        p1, p2 = chain[i], chain[i+1]
        if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
            return False
    return True

def _greedy_compact_init(n, rng):
    """Generate a compact chain using greedy nearest-neighbor to centroid."""
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for _ in range(n - 1):
        current = chain[-1]
        # Calculate centroid
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)
        cz = sum(p[2] for p in chain) / len(chain)
        
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if abs(dx) + abs(dy) + abs(dz) != 1:
                        continue
                    new_pos = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if new_pos not in occupied:
                        # Distance to centroid
                        dist = (new_pos[0]-cx)**2 + (new_pos[1]-cy)**2 + (new_pos[2]-cz)**2
                        candidates.append((dist, new_pos))
        
        if not candidates:
            # If stuck, break or try to backtrack? For init, just return what we have if too short, 
            # but we need exact length. If stuck, fallback to random walk from here.
            # Simple fallback: try random unoccupied neighbor of any previous node? No, must be connected.
            # If stuck, we fail this init and return None.
            return None
        
        # Pick the one with min distance to centroid (most compact)
        candidates.sort(key=lambda x: x[0])
        # Add some randomness: pick from top 3
        k = min(3, len(candidates))
        best_dist, best_pos = rng.choice(candidates[:k])
        chain.append(best_pos)
        occupied.add(best_pos)
        
    return chain

def _random_saw(n, rng):
    """Generate a random self-avoiding walk."""
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

def _pivot_move(chain, rng):
    """Perform a pivot move. Returns new chain or None if invalid."""
    n = len(chain)
    if n < 2:
        return None
    
    k = rng.randint(0, n - 2)
    pivot = chain[k]
    tail = chain[k+1:]
    
    # Rotate tail
    axis = rng.randint(0, 2)
    direction = rng.choice([-1, 1])
    
    def rotate(p):
        x, y, z = p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]
        if axis == 0:
            if direction == 1: return (pivot[0], pivot[1]+z, pivot[2]-y)
            else: return (pivot[0], pivot[1]-z, pivot[2]+y)
        elif axis == 1:
            if direction == 1: return (pivot[0]-z, pivot[1], pivot[2]+x)
            else: return (pivot[0]+z, pivot[1], pivot[2]-x)
        else:
            if direction == 1: return (pivot[0]+y, pivot[1]-x, pivot[2])
            else: return (pivot[0]-y, pivot[1]+x, pivot[2])
            
    new_tail = [rotate(p) for p in tail]
    new_chain = chain[:k+1] + new_tail
    
    if len(set(new_chain)) != n:
        return None
    # Check connectivity (pivot move preserves connectivity if rotation is valid 90 deg)
    # But we must ensure the first point of new_tail is adjacent to pivot.
    # Rotation of a vector (dx,dy,dz) by 90 deg around an axis results in a vector of same length.
    # Since original tail started with a unit vector, new tail starts with a unit vector.
    # So connectivity is preserved.
    return new_chain

def _local_search(sequence, chain, time_budget, rng):
    """Simulated Annealing local search."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 50.0
    t_final = 0.1
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        progress = min(1.0, elapsed / time_budget)
        temp = t0 + (t_final - t0) * progress
        
        # Choose move: 70% Pivot, 30% Creep (move last)
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, rng)
        else:
            # Creep move: move last residue
            i = n - 1
            prev = chain[i-1]
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if abs(dx) + abs(dy) + abs(dz) != 1:
                            continue
                        new_pos = (prev[0] + dx, prev[1] + dy, prev[2] + dz)
                        if new_pos == chain[i]:
                            continue
                        if new_pos in set(chain[:-1]):
                            continue
                        candidates.append(new_pos)
            if candidates:
                new_chain = chain[:-1] + [rng.choice(candidates)]
            else:
                new_chain = None
                
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold a protein sequence using Genetic Algorithm with Local Search.
    """
    n = len(sequence)
    rng = random.Random(98765 + len(sequence))
    
    # Time allocation
    # Total budget 240s for 26 sequences -> ~9s per sequence
    # We use 8.5s to be safe.
    time_budget_total = 8.5
    if n > 100:
        time_budget_total = 10.0
    if n < 50:
        time_budget_total = 6.0
        
    start_global = time.time()
    
    # 1. Initialization
    pop_size = 20
    population = []
    energies = []
    
    # Mix of Greedy Compact and Random SAW
    for i in range(pop_size):
        if i < pop_size // 2:
            chain = _greedy_compact_init(n, rng)
        else:
            chain = _random_saw(n, rng)
            
        if chain is None:
            # Fallback
            chain = [(i, 0, 0) for i in range(n)]
            
        energy = _compute_energy(sequence, chain)
        population.append(chain)
        energies.append(energy)
        
    # Sort by energy
    sorted_indices = sorted(range(pop_size), key=lambda i: energies[i])
    population = [population[i] for i in sorted_indices]
    energies = [energies[i] for i in sorted_indices]
    
    best_chain = population[0]
    best_energy = energies[0]
    
    # 2. Evolutionary Loop
    # Run local search on the best few individuals first to get a good baseline
    # Then evolve
    
    # Phase 1: Local Search on Top 5
    time_spent = time.time() - start_global
    remaining = time_budget_total - time_spent
    if remaining > 0:
        ls_time = remaining * 0.3
        for i in range(min(5, len(population))):
            time_left_for_ls = ls_time / (min(5, len(population)) - i)
            new_chain, new_energy = _local_search(sequence, population[i], time_left_for_ls, rng)
            population[i] = new_chain
            energies[i] = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
            
        # Re-sort
        sorted_indices = sorted(range(pop_size), key=lambda i: energies[i])
        population = [population[i] for i in sorted_indices]
        energies = [energies[i] for i in sorted_indices]
        
    # Phase 2: Genetic Algorithm
    time_spent = time.time() - start_global
    remaining = time_budget_total - time_spent
    if remaining > 0:
        generations = int(remaining / 0.2) # ~200ms per generation
        for g in range(generations):
            if time.time() - start_global > time_budget_total:
                break
                
            new_population = []
            new_energies = []
            
            # Keep best 2 (Elitism)
            new_population.append(population[0])
            new_energies.append(energies[0])
            new_population.append(population[1])
            new_energies.append(energies[1])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Selection: Tournament of 3
                p1_idx = rng.choice([0, 1, 2])
                p2_idx = rng.choice([0, 1, 2])
                p3_idx = rng.choice([0, 1, 2])
                
                # Pick best of 3
                best_idx = min([p1_idx, p2_idx, p3_idx], key=lambda i: energies[i])
                parent1 = population[best_idx]
                
                # Second parent: Random
                parent2 = population[rng.randint(0, pop_size-1)]
                
                # Crossover: Segment Swap
                # Pick random cut points
                if n > 4:
                    i1 = rng.randint(1, n-2)
                    i2 = rng.randint(1, n-2)
                    if i1 > i2:
                        i1, i2 = i2, i1
                    
                    child = list(parent1)
                    # Swap segment [i1:i2] with parent2's segment
                    # This might break connectivity.
                    # Simple approach: Just copy parent1 and mutate.
                    # Or: Use "Uniform Crossover" on moves?
                    # Let's stick to Mutation only for robustness, or a simple "Edge Swap" if valid.
                    # Given time constraints, Mutation is safer.
                    child = list(parent1)
                else:
                    child = list(parent1)
                    
                # Mutation: Pivot
                new_child = _pivot_move(child, rng)
                if new_child is None:
                    # Try another move
                    new_child = _pivot_move(child, rng)
                if new_child is None:
                    # Keep parent
                    new_child = child
                    
                new_energy = _compute_energy(sequence, new_child)
                new_population.append(new_child)
                new_energies.append(new_energy)
                
                # Update best
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_child
                    
            population = new_population
            energies = new_energies
            
            # Sort
            sorted_indices = sorted(range(pop_size), key=lambda i: energies[i])
            population = [population[i] for i in sorted_indices]
            energies = [energies[i] for i in sorted_indices]
            
    # Phase 3: Final Local Search on Best
    time_spent = time.time() - start_global
    remaining = time_budget_total - time_spent
    if remaining > 0.5:
        new_chain, new_energy = _local_search(sequence, best_chain, remaining, rng)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
