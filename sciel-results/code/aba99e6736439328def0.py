"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng, bias=0.5):
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
            # Weighted score
            score = bias * gain + (1 - bias) * free + rng.random() * 0.1
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

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

def _apply_pivot(chain, pivot, rotation):
    """
    Applies a pivot move.
    pivot: index in chain (1 to n-1). The bond is between pivot-1 and pivot.
    The segment [pivot, n-1] is rotated around chain[pivot-1].
    rotation: 1 (90 deg), -1 (-90 deg), 2 (180 deg).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if pivot <= 0 or pivot >= n:
        return None
        
    p_pivot = chain[pivot-1]
    v = (chain[pivot][0] - p_pivot[0], chain[pivot][1] - p_pivot[1])
    
    if rotation == 1:
        v_rot = (-v[1], v[0])
    elif rotation == -1:
        v_rot = (v[1], -v[0])
    elif rotation == 2:
        v_rot = (-v[0], -v[1])
    else:
        return None
        
    new_chain = [None] * n
    for i in range(pivot):
        new_chain[i] = chain[i]
        
    base_new = (p_pivot[0] + v_rot[0], p_pivot[1] + v_rot[1])
    
    # Occupied positions in the fixed part (0 to pivot-2)
    # Note: chain[pivot-1] is the pivot point, it is fixed.
    # The new segment starts at base_new.
    # base_new must not collide with 0..pivot-2.
    occupied_fixed = set(chain[:pivot-1])
    
    if base_new in occupied_fixed:
        return None
        
    occupied_rot = {base_new}
    new_chain[pivot] = base_new
    
    for k in range(pivot + 1, n):
        offset = (chain[k][0] - chain[pivot][0], chain[k][1] - chain[pivot][1])
        new_pos = (base_new[0] + offset[0], base_new[1] + offset[1])
        
        if new_pos in occupied_fixed or new_pos in occupied_rot:
            return None
        occupied_rot.add(new_pos)
        new_chain[k] = new_pos
        
    return new_chain

def _local_search(sequence, chain, energy, deadline):
    n = len(chain)
    best_chain = chain[:]
    best_energy = energy
    current_chain = chain[:]
    current_energy = energy
    
    stagnation = 0
    max_stagnation = 100
    
    while time.time() < deadline:
        if n < 2:
            break
            
        pivot = random.randint(1, n-1)
        rotation = random.choice([1, -1, 2])
        
        new_chain = _apply_pivot(current_chain, pivot, rotation)
        
        if new_chain is None:
            stagnation += 1
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            stagnation = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                # Kick: try to escape local minimum
                # Accept a random valid move
                escaped = False
                for _ in range(20):
                    p = random.randint(1, n-1)
                    r = random.choice([1, -1, 2])
                    kick_chain = _apply_pivot(current_chain, p, r)
                    if kick_chain is not None:
                        current_chain = kick_chain
                        current_energy = _energy(sequence, kick_chain)
                        stagnation = 0
                        escaped = True
                        break
                if not escaped:
                    break

    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    # 120s total for 10 sequences. Use 11.5s per sequence to be safe.
    deadline = start_time + 11.5
    
    # 1. Generate diverse initial chains
    initial_chains = []
    # Different biases to explore different parts of the conformational space
    biases = [0.8, 0.6, 0.4, 0.2, 0.0, 1.0, 0.7, 0.3, 0.5, 0.9]
    
    # Generate 50 initial chains
    for i in range(50):
        if time.time() > deadline - 1.0: # Stop generating if close to deadline
            break
        bias = biases[i % len(biases)]
        rng = random.Random(12345 + i + len(sequence))
        chain = _grow(sequence, dimension, rng, bias)
        if chain:
            energy = _energy(sequence, chain)
            initial_chains.append((energy, chain))
            
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
        
    # Sort by energy
    initial_chains.sort(key=lambda x: x[0])
    
    # Take top 5 for local search
    top_chains = initial_chains[:5]
    
    best_chain = top_chains[0][1]
    best_energy = top_chains[0][0]
    
    # 2. Local Search on top chains
    for energy, chain in top_chains:
        if time.time() > deadline - 0.5: # Leave 0.5s for return
            break
            
        improved_chain, improved_energy = _local_search(sequence, chain, energy, deadline)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
