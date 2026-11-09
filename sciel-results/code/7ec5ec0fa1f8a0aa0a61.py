"""
Optimized HP Lattice Protein Folding using Simulated Annealing.
"""
import random
import time
import math

# Global deadline guard
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0  # Safety margin before 330s kill switch
_DEADLINE = _START_TIME + _TOTAL_BUDGET

def _neighbours(point):
    """Yield the 6 neighbors of a point on the cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _initial_grow(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using a biased random walk.
    This provides a valid starting point for the annealing process.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    # Bias towards compacting the chain
    # We track the center of mass to bias growth towards it
    com = [0.0, 0.0, 0.0]
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        for n in _neighbours(last):
            if n not in occupied:
                # Calculate distance to center of mass
                dist = sum((n[j] - com[j])**2 for j in range(3))
                # Add some randomness
                candidates.append((dist + rng.random() * 5, n))
        
        if not candidates:
            # Stuck, try to backtrack or just fail (rare in 3D for reasonable lengths)
            # For simplicity, if stuck, we return None and fold() will try again
            return None
        
        # Sort by distance to COM (compactness)
        candidates.sort(key=lambda x: x[0])
        
        # Pick from the top few to maintain some diversity
        # For initial growth, we can be greedy
        site = candidates[0][1]
        
        chain.append(site)
        occupied.add(site)
        
        # Update center of mass
        for j in range(3):
            com[j] = (com[j] * i + site[j]) / (i + 1)
            
    return chain

def _calculate_energy(sequence, chain):
    """Calculate the total energy of a chain."""
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n in _neighbours(p):
                j = pos_to_idx.get(n)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    # Count each contact twice (once for i, once for j), so divide by 2 at end
                    # Or just count unique pairs. Let's count unique pairs to be safe.
                    if i < j:
                        contacts += 1
    return -contacts

def _apply_move(chain, move_type, start, end, new_positions=None):
    """
    Apply a move to the chain.
    move_type: 'kink' or 'tail'
    For kink: reverse segment chain[start:end+1]
    For tail: replace chain[start:] with new_positions
    Returns the new chain.
    """
    if move_type == 'kink':
        # Reverse segment
        new_chain = chain[:start] + list(reversed(chain[start:end+1])) + chain[end+1:]
        return new_chain
    elif move_type == 'tail':
        # Replace tail
        new_chain = chain[:start] + new_positions
        return new_chain
    else:
        return chain

def _calculate_delta_energy(sequence, old_chain, new_chain, start, end, move_type):
    """
    Calculate the change in energy between old_chain and new_chain.
    Only residues in the affected region need to be checked.
    """
    n = len(sequence)
    # Build position to index map for old_chain
    old_pos_to_idx = {p: i for i, p in enumerate(old_chain)}
    # Build position to index map for new_chain
    new_pos_to_idx = {p: i for i, p in enumerate(new_chain)}
    
    # Identify indices that are affected
    # For kink: indices start..end
    # For tail: indices start..n-1
    # We need to check contacts involving these indices with ALL other indices
    # But only those that are H-H.
    
    affected_indices = set(range(start, end + 1))
    
    delta = 0
    # Remove old contacts involving affected indices
    for i in affected_indices:
        if sequence[i] == 'H':
            p_old = old_chain[i]
            for n in _neighbours(p_old):
                j = old_pos_to_idx.get(n)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    # This contact existed in old chain
                    # We subtract it. To avoid double counting in delta, 
                    # we can just count all old contacts involving affected set 
                    # and all new contacts involving affected set.
                    # Since we are calculating delta = E_new - E_old, 
                    # and E = -contacts, delta = -(C_new - C_old) = C_old - C_new.
                    # Let's just count C_old_part and C_new_part involving affected indices.
                    pass
                    
    # A simpler and more robust way for moderate N:
    # Just recalculate energy for the whole chain? 
    # N is up to 136. O(N^2) is ~18k ops. 
    # With 1000s of moves, this is fine.
    # Let's stick to full recalculation for simplicity and correctness, 
    # but optimize the neighbor lookup.
    
    e_old = _calculate_energy(sequence, old_chain)
    e_new = _calculate_energy(sequence, new_chain)
    return e_new - e_old

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Time allocation based on length
    # Longer chains need more time
    time_alloc = min(10.0, n * 0.1)  # Cap at 10s per sequence to be safe
    deadline = time.time() + time_alloc
    
    rng = random.Random(42 + n)
    
    # Initial conformation
    best_chain = None
    best_energy = 0
    
    # Try to get a good initial chain
    attempts = 0
    while time.time() < deadline and attempts < 10:
        chain = _initial_grow(sequence, dimension, rng)
        if chain is not None:
            energy = _calculate_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
            break
        attempts += 1
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _calculate_energy(sequence, best_chain)
        
    current_chain = best_chain
    current_energy = best_energy
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    # Pre-compute neighbor list for speed? No, _neighbours is fast enough.
    
    while time.time() < deadline and T > min_T:
        # Choose a move
        # Kink move: pick start, end such that end - start >= 2
        # Tail move: pick start, regrow
        if n > 10 and rng.random() < 0.5:
            # Kink
            length = rng.randint(2, min(n // 2, 20))
            start = rng.randint(0, n - length - 1)
            end = start + length
            # Check if kink is valid (no self-intersection)
            new_chain = _apply_move(current_chain, 'kink', start, end)
            
            # Check self-avoidance
            if len(set(new_chain)) == len(new_chain):
                delta_e = _calculate_delta_energy(sequence, current_chain, new_chain, start, end, 'kink')
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_chain = new_chain
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain
        else:
            # Tail move
            tail_len = rng.randint(2, min(n // 2, 15))
            start = n - tail_len
            # Regrow tail
            prefix = current_chain[:start]
            occupied = set(prefix)
            last = prefix[-1]
            
            new_tail = []
            valid = True
            for _ in range(tail_len):
                candidates = [n for n in _neighbours(last) if n not in occupied]
                if not candidates:
                    valid = False
                    break
                # Randomly pick a candidate, maybe bias towards compactness
                # For simplicity, random
                site = rng.choice(candidates)
                new_tail.append(site)
                occupied.add(site)
                last = site
                
            if valid:
                new_chain = prefix + new_tail
                # Self-avoidance is guaranteed by construction of new_tail
                # But we must ensure new_tail doesn't intersect prefix (already checked in occupied)
                # And new_tail doesn't intersect itself (checked in occupied)
                delta_e = _calculate_delta_energy(sequence, current_chain, new_chain, start, n-1, 'tail')
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_chain = new_chain
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain
                        
        T *= cooling
        
        # Check global deadline
        if time.time() > _DEADLINE - 5.0:
            break
            
    return [list(p) for p in best_chain]
