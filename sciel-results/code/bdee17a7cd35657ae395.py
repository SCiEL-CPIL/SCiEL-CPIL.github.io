"""HP lattice protein folding on the cubic lattice - Simulated Annealing with End-Moves and Pivot Moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _compute_energy_full(sequence, chain):
    """Compute energy of the chain from scratch. O(N)."""
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

def _initial_greedy_saw(sequence, rng):
    """Generate a self-avoiding walk with a greedy bias towards H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for i in range(1, n):
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
            # Fallback to random SAW if stuck
            return _initial_random_saw(sequence, rng)
        
        if sequence[i] == 'H':
            # Greedy: prefer positions that are neighbors of other H's
            best_score = -1
            best_pos = None
            for pos in candidates:
                score = 0
                for q in [(pos[0]+1, pos[1], pos[2]), (pos[0]-1, pos[1], pos[2]), 
                          (pos[0], pos[1]+1, pos[2]), (pos[0], pos[1]-1, pos[2]), 
                          (pos[0], pos[1], pos[2]+1), (pos[0], pos[1], pos[2]-1)]:
                    if q in occupied:
                        # Check if the occupant is an H and not the previous residue
                        idx = chain.index(q) if q in chain else -1
                        if idx != -1 and idx < i - 1 and sequence[idx] == 'H':
                            score += 1
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos and best_score > 0:
                # With some probability, take the greedy step
                if rng.random() < 0.8:
                    next_pos = best_pos
                else:
                    next_pos = rng.choice(candidates)
            else:
                next_pos = rng.choice(candidates)
        else:
            next_pos = rng.choice(candidates)
        
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain

def _initial_random_saw(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    for i in range(1, n):
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

def _rotate_90(point, axis, direction):
    x, y, z = point
    if axis == 0:
        if direction == 1: return (x, z, -y)
        else: return (x, -z, y)
    elif axis == 1:
        if direction == 1: return (-z, y, x)
        else: return (z, y, -x)
    else:
        if direction == 1: return (y, -x, z)
        else: return (-y, x, z)

def _sa_fold(sequence, dimension, time_budget, rng):
    n = len(sequence)
    
    # Initialization
    chain = _initial_greedy_saw(sequence, rng)
    if chain is None:
        chain = _initial_random_saw(sequence, rng)
    if chain is None:
        chain = [(i, 0, 0) for i in range(n)]
    
    # Build occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy_full(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 20.0
    t_final = 0.5
    t_half = time_budget / 4.0 # Cool down to half temp halfway? Or linear?
    
    # We will use a linear cooling schedule based on time
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        progress = min(1.0, elapsed / time_budget)
        temp = t0 * (t_final / t0) ** progress
        
        # Choose move: 90% End Move, 10% Pivot Move
        if rng.random() < 0.9:
            # End Move: Move the last residue
            if n < 2:
                continue
            last_idx = n - 1
            prev_pos = chain[last_idx - 1]
            last_pos = chain[last_idx]
            
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if abs(dx) + abs(dy) + abs(dz) != 1:
                            continue
                        new_pos = (prev_pos[0] + dx, prev_pos[1] + dy, prev_pos[2] + dz)
                        if new_pos == last_pos:
                            continue
                        # Check collision with chain[0:-1]
                        # We need to check if new_pos is in occupied and the occupant is not the last residue itself
                        if new_pos in occupied:
                            if occupied[new_pos] == last_idx:
                                continue # This is the residue itself moving away
                            else:
                                continue # Collision
                        candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Calculate Energy Delta for End Move
            # Remove last_pos from occupied
            del occupied[last_pos]
            occupied[new_pos] = last_idx
            
            # Recalculate contacts for the last residue
            # Old contacts: neighbors of last_pos that are H and index < last_idx - 1
            # New contacts: neighbors of new_pos that are H and index < last_idx - 1
            
            old_contacts = 0
            if sequence[last_idx] == 'H':
                for q in [(last_pos[0]+1, last_pos[1], last_pos[2]), (last_pos[0]-1, last_pos[1], last_pos[2]), 
                          (last_pos[0], last_pos[1]+1, last_pos[2]), (last_pos[0], last_pos[1]-1, last_pos[2]), 
                          (last_pos[0], last_pos[1], last_pos[2]+1), (last_pos[0], last_pos[1], last_pos[2]-1)]:
                    j = occupied.get(q)
                    if j is not None and j < last_idx - 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            new_contacts = 0
            if sequence[last_idx] == 'H':
                for q in [(new_pos[0]+1, new_pos[1], new_pos[2]), (new_pos[0]-1, new_pos[1], new_pos[2]), 
                          (new_pos[0], new_pos[1]+1, new_pos[2]), (new_pos[0], new_pos[1]-1, new_pos[2]), 
                          (new_pos[0], new_pos[1], new_pos[2]+1), (new_pos[0], new_pos[1], new_pos[2]-1)]:
                    j = occupied.get(q)
                    if j is not None and j < last_idx - 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            delta_e = -(new_contacts - old_contacts)
            
            # Metropolis
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
                chain[last_idx] = new_pos
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Reject: revert occupied
                del occupied[new_pos]
                occupied[last_pos] = last_idx
        
        else:
            # Pivot Move: Rotate tail
            if n < 2:
                continue
            k = rng.randint(0, n - 2)
            pivot = chain[k]
            tail = chain[k+1:]
            
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            
            tail_rel = [(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]) for p in tail]
            tail_rot = [_rotate_90(p, axis, direction) for p in tail_rel]
            new_tail = [(p[0] + pivot[0], p[1] + pivot[1], p[2] + pivot[2]) for p in tail_rot]
            
            # Check validity
            fixed_set = set(chain[:k+1])
            new_chain = chain[:k+1] + new_tail
            new_occupied = set(new_chain)
            
            if len(new_occupied) != n:
                continue
            
            # Compute new energy (Expensive, but Pivot is rare)
            new_energy = _compute_energy_full(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
                chain = new_chain
                occupied = {p: i for i, p in enumerate(chain)}
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using Simulated Annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(54321 + len(sequence))
    
    # Time allocation
    # Base 2s, + 2s per 100 residues, max 15s
    time_budget = 2.0 + (n / 100.0) * 2.0
    time_budget = min(time_budget, 15.0)
    
    chain, energy = _sa_fold(sequence, dimension, time_budget, rng)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
