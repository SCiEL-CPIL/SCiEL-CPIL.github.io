"""HP lattice protein folding on cubic lattice - beam search + local refinement."""
# EVOLVE-BLOCK-START
import time
import random
import math

START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a full chain (negative number of H-H contacts)."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50, time_budget=15.0):
    """
    Beam search for HP folding.
    Returns best full chain or None if timeout.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    deadline = START_TIME + time_budget
    
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_energy)]
    
    for i in range(1, n):
        next_beam = []
        
        for chain, occupied, energy in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_energy -= 1
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_energy))
        
        if not next_beam:
            return None
        
        # Prune to beam_width
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        # Time check
        if time.time() > deadline:
            return None
    
    if not beam:
        return None
        
    return beam[0][0]

def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy growth."""
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
            score = gain * 2.0 + free * 0.3 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            break
        
        options.sort(reverse=True)
        # Pick from top few with some randomness
        top_k = min(3, len(options))
        chosen = random.choice(options[:top_k])[1]
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain

def _local_refine(sequence, chain, dimension, max_iters=50):
    """
    Local refinement: try to improve energy by moving individual residues.
    A move is valid if:
    1. The new position is unoccupied
    2. The chain remains connected (consecutive residues are adjacent)
    3. The chain remains self-avoiding
    """
    n = len(chain)
    if n == 0:
        return chain, 0
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iters):
        improved = False
        occupied = {p: i for i, p in enumerate(best_chain)}
        
        for i in range(n):
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
            
            pos = best_chain[i]
            
            # Try moving residue i to a neighboring empty site
            for new_pos in _neighbours(pos):
                if new_pos in occupied:
                    continue
                
                # Check if this move maintains connectivity
                # The residue at position i must be adjacent to i-1 and i+1 (if they exist)
                valid = True
                
                if i > 0:
                    # Must be adjacent to previous residue
                    prev_pos = best_chain[i-1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, prev_pos))
                    if dist != 1:
                        valid = False
                
                if valid and i < n-1:
                    # Must be adjacent to next residue
                    next_pos = best_chain[i+1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, next_pos))
                    if dist != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Tentatively apply the move
                old_pos = best_chain[i]
                best_chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                # Check self-avoidance
                if len(occupied) != n:
                    # Collision detected, revert
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
                    continue
                
                # Compute new energy
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                else:
                    # Revert
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
            
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using beam search with local refinement and multiple restarts.
    """
    n = len(sequence)
    
    # Determine time budget for beam search based on sequence length
    if n <= 48:
        beam_time = 20.0
        beam_width = 100
    elif n <= 80:
        beam_time = 15.0
        beam_width = 70
    else:
        beam_time = 10.0
        beam_width = 50
    
    # Try beam search first
    best_chain = _beam_search(sequence, dimension, beam_width, beam_time)
    
    if best_chain is None or len(best_chain) != n:
        # Fallback to greedy growth with randomization
        rng = random.Random(12345 + n)
        best_chain = _greedy_grow(sequence, dimension, rng)
        
        if len(best_chain) != n:
            # Last resort: straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Local refinement
    best_chain, _ = _local_refine(sequence, best_chain, dimension, max_iters=30)
    
    # If we have time, try multiple greedy starts and keep the best
    if time.time() - START_TIME < TIME_LIMIT * 0.6:
        best_energy = _compute_energy(sequence, best_chain)
        rng = random.Random(99999 + n)
        
        num_restarts = min(5, int((TIME_LIMIT * 0.6 - (time.time() - START_TIME)) / 2))
        
        for restart in range(max(1, num_restarts)):
            if time.time() - START_TIME > TIME_LIMIT * 0.8:
                break
            
            candidate = _greedy_grow(sequence, dimension, rng)
            if len(candidate) != n:
                continue
            
            candidate, cand_energy = _local_refine(sequence, candidate, dimension, max_iters=20)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_chain = candidate
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
