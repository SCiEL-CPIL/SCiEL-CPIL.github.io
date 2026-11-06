"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 185s as a safe deadline to ensure we finish before kill.
    return 185.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy(sequence, chain):
    """Compute energy: minus number of H-H contacts."""
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
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbors for diversity
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain * 10.0 + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        # Select best option
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, max_steps=5000, rng=None):
    """
    Lightweight local search: try to improve energy by moving individual residues
    or performing small chain reversals (2-opt-like moves).
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute which positions are H
    is_h = [1 if c == 'H' else 0 for c in sequence]
    
    # Try to improve by local moves
    # Move 1: Try to move a single residue to an adjacent empty site
    # Move 2: Try to reverse a subchain (like 2-opt in TSP)
    
    for step in range(max_steps):
        if step % 100 == 0 and _time_remaining() < 5.0:
            break
        
        improved = False
        
        # Try moving a random residue (not endpoints) to an adjacent empty site
        if n > 2:
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            # Try each neighbor of old_pos
            for new_pos in _neighbours(old_pos):
                # Check if new_pos is occupied by a non-adjacent residue
                if new_pos in where:
                    j = where[new_pos]
                    if abs(i - j) > 1:
                        continue  # Would cause collision
                
                # Check if new_pos is free or occupied by adjacent
                if new_pos in where:
                    j = where[new_pos]
                    if abs(i - j) != 1:
                        continue
                
                # Tentatively move
                # Remove old_pos from where
                del where[old_pos]
                # Add new_pos to where
                where[new_pos] = i
                chain[i] = new_pos
                
                # Check self-avoidance: new_pos must not collide with non-adjacent residues
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    if chain[j] == new_pos:
                        # Check if j is adjacent to i
                        if abs(j - i) > 1:
                            valid = False
                            break
                
                if not valid:
                    # Revert
                    where[new_pos] = where[old_pos] if old_pos in where else None
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
                    continue
                
                # Compute new energy
                new_energy = _energy(sequence, chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                else:
                    # Revert
                    where[new_pos] = where[old_pos] if old_pos in where else None
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
        
        if not improved and step > 100:
            # Try a 2-opt reversal
            if n > 4:
                i = rng.randint(1, n - 3)
                j = rng.randint(i + 1, n - 2)
                
                # Reverse chain[i:j+1]
                sub = chain[i:j+1][::-1]
                old_sub = chain[i:j+1]
                
                # Check if reversal is valid (no self-avoidance violations)
                valid = True
                new_where = {}
                for k in range(n):
                    if i <= k <= j:
                        new_where[sub[k - i]] = k
                    else:
                        new_where[chain[k]] = k
                
                # Check for collisions
                seen = set()
                for pos in new_where:
                    if pos in seen:
                        valid = False
                        break
                    seen.add(pos)
                
                if valid:
                    # Apply reversal
                    chain[i:j+1] = sub
                    where = new_where
                    
                    new_energy = _energy(sequence, chain)
                    
                    if new_energy < current_energy:
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        improved = True
                    else:
                        # Revert
                        chain[i:j+1] = old_sub
                        where = {p: i for i, p in enumerate(chain)}
        
        if not improved and step > 500:
            break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search."""
    rng = random.Random(12345 + len(sequence))
    
    # Number of restarts based on sequence length
    n = len(sequence)
    if n <= 30:
        num_restarts = 500
    elif n <= 50:
        num_restarts = 300
    elif n <= 80:
        num_restarts = 150
    else:
        num_restarts = 80
    
    best_chain = None
    best_energy = 1  # Start with unfolded (energy 0), but we want negative
    
    for restart in range(num_restarts):
        if _time_remaining() < 2.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        local_rng = random.Random(restart * 1000 + hash(sequence) % 10000)
        improved_chain = _local_search(sequence, chain, max_steps=2000, rng=local_rng)
        
        value = _energy(sequence, improved_chain)
        if value < best_energy:
            best_energy = value
            best_chain = improved_chain
        
        # If we found a very good solution, we can stop early
        # Heuristic: for HP sequences, the minimum energy is roughly -n/4 to -n/3
        if best_energy <= -n // 4:
            break
    
    if best_chain is None:
        # Fallback: straight line (always self-avoiding)
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
