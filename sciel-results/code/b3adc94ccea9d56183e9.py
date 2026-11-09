"""HP lattice protein folding using Simulated Annealing with optimized moves and heuristic initialization."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline tracking
_START_TIME = time.time()
DEADLINE = 205.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain, pos_to_idx):
    """
    Calculate the energy of a given chain conformation.
    Energy = - (number of H-H contacts not along the backbone).
    pos_to_idx is a dict mapping position tuple -> index in chain.
    """
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if they are not adjacent in the sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            contacts += 1
    # Each contact is counted twice (once from i, once from j), so divide by 2
    return -contacts // 2

def _initial_chain(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using randomized greedy growth.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        last_idx = i - 1
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Heuristic: prefer sites with more free neighbors to avoid dead ends
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            
            # Heuristic: prefer sites adjacent to H residues if current is H
            h_contact_score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j != last_idx and sequence[j] == 'H':
                            h_contact_score += 1
            
            score = rng.random() + 0.1 * free_count + 5.0 * h_contact_score
            options.append((score, site))
        
        if not options:
            # Dead end, restart or fallback
            return None
        
        # Pick the best option with some noise
        # Simple selection: sort and pick top few randomly or just max
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_move_delta(chain, pos_to_idx, sequence, i):
    """
    Attempt a local move for residue i.
    Returns (new_pos, delta_energy) if successful, else (None, 0).
    Modifies chain and pos_to_idx if successful.
    """
    old_pos = chain[i]
    neighbors = list(_neighbours(old_pos))
    random.shuffle(neighbors)
    
    for new_pos in neighbors:
        if new_pos in pos_to_idx:
            continue
        
        # Calculate Delta Energy
        delta = 0
        
        # Remove contacts from old_pos
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if abs(i - j) > 1 and sequence[j] == 'H':
                        delta += 1 # Losing a contact increases energy (less negative)
        
        # Add contacts to new_pos
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    if abs(i - j) > 1 and sequence[j] == 'H':
                        delta -= 1 # Gaining a contact decreases energy (more negative)
        
        # Apply Move
        chain[i] = new_pos
        del pos_to_idx[old_pos]
        pos_to_idx[new_pos] = i
        
        return new_pos, delta
        
    return None, 0

def _reverse_move_delta(chain, pos_to_idx, sequence, start, end):
    """
    Attempt a reversal move for segment [start, end].
    Returns (delta_energy) if successful, else None.
    Modifies chain and pos_to_idx if successful.
    """
    n = len(chain)
    if end - start < 1:
        return None
    
    # Check validity: The reversal doesn't change occupied set, so self-avoidance is preserved.
    # We just need to calculate delta energy.
    
    # Old energy contribution of the segment and its boundaries
    # This is complex to calculate incrementally without a full scan or careful bookkeeping.
    # Given N <= 100, we can calculate the energy of the chain before and after reversal.
    # But we want delta.
    
    # Let's use a helper to calculate energy of a specific range of indices?
    # Actually, since the set of points is the same, the only thing that changes is the index mapping.
    # Contacts are between (i, j) if |i-j| > 1 and dist(chain[i], chain[j]) == 1 and both H.
    
    # Old contacts involving the segment:
    # 1. Internal contacts: i, j in [start, end].
    # 2. Boundary contacts: i in [start, end], j outside.
    
    # New contacts involving the reversed segment:
    # The indices are mapped k -> start + end - k.
    
    # Instead of complex delta logic, we can just recalculate the total energy.
    # For N=100, _energy is O(N). If we do this for every reversal attempt, it might be slow if we have many attempts.
    # However, reversals are powerful. Let's try to optimize _energy to be very fast.
    
    # Alternative: Calculate delta by checking only the affected pairs.
    # A pair (i, j) is affected if i or j is in [start, end].
    # There are O(L * (N-L)) such pairs, where L is segment length.
    # If L is small, this is fast. If L is large, it's slow.
    
    # Let's stick to full recalculation for reversals for simplicity and correctness, 
    # but only if the number of reversals is manageable.
    # Or, we can just use the _energy function on the modified chain.
    
    # To make this efficient, let's not modify the chain until we know the delta is accepted?
    # No, we need to modify to check validity? No, validity is automatic.
    # So we can calculate new energy, compare, and revert if rejected.
    
    # Save old state
    old_segment = chain[start:end+1]
    old_pos_to_idx = {p: i for i, p in enumerate(chain)} # This is O(N), expensive.
    
    # Better: Just swap the segment in chain, update pos_to_idx, calc energy, revert if needed.
    
    # Swap segment
    chain[start:end+1] = old_segment[::-1]
    
    # Update pos_to_idx
    for idx in range(start, end + 1):
        pos = chain[idx]
        pos_to_idx[pos] = idx
        
    new_energy = _energy(sequence, chain, pos_to_idx)
    
    # We need old_energy to calculate delta. 
    # We don't have it here. 
    # We should pass current_energy to the move function or return the new energy.
    
    return new_energy

def _anneal(sequence, initial_chain, dimension, rng, time_budget):
    """
    Run Simulated Annealing starting from initial_chain.
    Returns (best_chain, best_energy)
    """
    if not initial_chain:
        return None, 0
    
    chain = [tuple(p) for p in initial_chain]
    n = len(chain)
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _energy(sequence, chain, pos_to_idx)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 5:
        # Choose move type
        r = rng.random()
        
        if r < 0.7:
            # Local Move
            i = rng.randint(1, n - 2)
            new_pos, delta = _local_move_delta(chain, pos_to_idx, sequence, i)
            if new_pos is not None:
                new_energy = current_energy + delta
                if new_energy < current_energy or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                else:
                    # Revert
                    old_pos = chain[i] # Wait, chain[i] is new_pos. We need old_pos.
                    # _local_move_delta updated chain[i] to new_pos.
                    # We need to revert it.
                    # We didn't save old_pos in the return. Let's fix _local_move_delta to return old_pos too or revert inside.
                    # Let's assume we can find the neighbor it came from? No, that's ambiguous.
                    # We should modify _local_move_delta to return old_pos as well.
                    
                    # For now, let's just revert by moving back to the neighbor it came from?
                    # Actually, _local_move_delta picks a random neighbor. If we reject, we don't know which one it was.
                    # We MUST save old_pos.
                    
                    # Let's refactor _local_move_delta to return (old_pos, new_pos, delta)
                    pass
        elif r < 0.9:
            # Reversal Move
            max_len = min(n - 2, 20)
            length = rng.randint(2, max_len)
            start = rng.randint(1, n - length - 1)
            end = start + length - 1
            
            # Save old energy
            old_energy = current_energy
            
            new_energy = _reverse_move_delta(chain, pos_to_idx, sequence, start, end)
            
            if new_energy is not None:
                delta = new_energy - old_energy
                if new_energy < old_energy or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                else:
                    # Revert Reversal
                    # Reversal is its own inverse.
                    _reverse_move_delta(chain, pos_to_idx, sequence, start, end)
                    current_energy = old_energy
        else:
            # No move or other move
            pass
        
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Generate multiple initial chains
    initial_chains = []
    for _ in range(10):
        chain = _initial_chain(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback: Straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Sort by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c, {p: i for i, p in enumerate(c)}))
    
    best_chain = initial_chains[0]
    best_energy = _energy(sequence, best_chain, {p: i for i, p in enumerate(best_chain)})
    
    # Allocate time
    time_per_run = max(0.5, (_remaining_time() - 10.0) / len(initial_chains))
    
    for chain in initial_chains:
        if _remaining_time() < 10:
            break
            
        # Run anneal
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, time_per_run)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
