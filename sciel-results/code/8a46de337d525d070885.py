"""Rewritten program for HP lattice protein folding using greedy growth followed by simulated annealing polish."""
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

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _calculate_energy(sequence, chain):
    occupied = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def _polish(chain, sequence, dimension, rng, deadline):
    """Simulated annealing polish on the chain."""
    N = len(chain)
    if N < 5:
        return chain, _calculate_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _calculate_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA parameters
    T = 1.0
    cooling = 0.99
    steps_per_temp = 20
    
    while T > 0.01 and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() >= deadline:
                break
            
            # Pick a random residue to move (1 to N-2)
            i = rng.randint(1, N - 2)
            old_pos = chain[i]
            
            # Pick a random neighbor
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            
            for new_pos in neighbors:
                # Check if new_pos is free
                if new_pos in occupied:
                    if occupied[new_pos] != i:
                        continue # Collision
                    else:
                        continue # Same spot
                
                # Calculate delta energy
                delta = 0
                if sequence[i] == 'H':
                    # Remove contacts from old_pos
                    for q in _neighbours(old_pos):
                        if q in occupied:
                            j = occupied[q]
                            if j != i and sequence[j] == 'H':
                                delta += 1
                    # Add contacts to new_pos
                    for q in _neighbours(new_pos):
                        if q in occupied:
                            j = occupied[q]
                            if j != i and sequence[j] == 'H':
                                delta -= 1
                
                # Acceptance criterion
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    # Perform move
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break
        T *= cooling
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths + polish."""
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _calculate_energy(sequence, best_chain)
        
    while time.time() < deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Polish with the global deadline
        polished_chain, energy = _polish(chain, sequence, dimension, rng, deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = polished_chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
