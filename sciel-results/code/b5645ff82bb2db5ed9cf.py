"""
HP Lattice Protein Folding on Cubic Lattice
Uses Monte Carlo Chain Growth (MCCG) with contact-based energy bias and local search.
"""
import math

import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y + 1, z) # Duplicate in original? No, just listing. 
    # Wait, original had:
    # yield (x, y + 1, z)
    # yield (x, y - 1, z)
    # yield (x, y, z + 1)
    # yield (x, y, z - 1)
    # My list above was:
    # (x+1, y, z)
    # (x-1, y, z)
    # (x, y+1, z)
    # (x, y-1, z)
    # (x, y, z+1)
    # (x, y, z-1)
    # Let's stick to the standard 6.
    yield (x, y, z + 1)
    yield (x, y, z - 1)

# Note: The original _neighbours had a duplicate or error? 
# Original:
# yield (x + 1, y, z)
# yield (x - 1, y, z)
# yield (x, y + 1, z)
# yield (x, y - 1, z)
# yield (x, y, z + 1)
# yield (x, y, z - 1)
# This is correct. I will keep it identical to avoid breaking anything relying on order, though order shouldn't matter for sets.

def _count_hh_contacts(sequence, chain):
    """
    Count the number of H-H contacts in the chain.
    A contact is between two H residues that are on neighboring lattice sites
    but not adjacent along the chain (|i - j| > 1).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    occ = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n_pos in _neighbours(p):
                j = occ.get(n_pos)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    # Each contact is counted twice (once from each H), so divide by 2
    return contacts // 2

def _calculate_energy(sequence, chain):
    """Calculate total energy of a chain. Energy = -number of H-H contacts."""
    return -_count_hh_contacts(sequence, chain)

def _calculate_energy_delta_grow(sequence, index, pos, occupied):
    """
    Calculate the change in energy if residue at 'index' is placed at 'pos'
    during growth. Only checks neighbors that are already in 'occupied'.
    """
    if sequence[index] != 'H':
        return 0
    
    delta = 0
    for n_pos in _neighbours(pos):
        if n_pos in occupied:
            n_idx = occupied[n_pos]
            # Check if not adjacent in sequence
            if abs(index - n_idx) > 1:
                if sequence[n_idx] == 'H':
                    delta -= 1  # Energy decreases (becomes more negative)
    return delta

def _grow_contact_biased(sequence, rng, T=1.0, contact_weight=3.0):
    """
    Grow a self-avoiding walk with strong bias toward creating H-H contacts.
    Uses a contact-based scoring function.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = list(_neighbours(last))
        # Filter out occupied
        available = [p for p in neighbors if p not in occupied]
        
        if not available:
            return None
        
        # Calculate score for each available position
        scores = []
        for p in available:
            # Immediate contact gain
            dE = _calculate_energy_delta_grow(sequence, i, p, occupied)
            
            # Heuristic: Prefer positions that are close to other H residues
            # that are not yet adjacent in the chain but might become so?
            # Actually, simple contact count is usually sufficient for MCCG if weighted well.
            # Let's add a small penalty for "dead ends" or "tight" spots if possible, 
            # but standard MCCG relies on the energy bias.
            
            # To improve on "local gains", we can look at the number of free neighbors
            # to avoid trapping the chain.
            free_neighbors = 0
            for n_pos in _neighbours(p):
                if n_pos not in occupied and n_pos != last:
                    free_neighbors += 1
            
            # Score = -dE (positive for good contacts) * weight + free_neighbors * small_bonus
            # dE is negative for favorable contacts.
            immediate_score = max(0, -dE) * contact_weight
            flexibility_score = free_neighbors * 0.2
            
            total_score = immediate_score + flexibility_score
            
            # Boltzmann factor
            weight = math.exp(total_score / T)
            scores.append((weight, p))
        
        # Normalize and pick
        total_w = sum(w for w, _ in scores)
        if total_w == 0:
            chosen = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumulative = 0
            chosen = scores[0][1]
            for w, p in scores:
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
        
    return chain

def _local_search_contacts(sequence, chain, rng, steps=500):
    """
    Apply local search moves to optimize the chain by maximizing H-H contacts.
    Uses pivot moves (rotating a segment) and reversal moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    c = chain[:]
    occ = {p: i for i, p in enumerate(c)}
    
    current_contacts = _count_hh_contacts(sequence, c)
    best_chain = c[:]
    best_contacts = current_contacts
    
    T = 1.0
    cooling = 0.005
    min_T = 0.05
    
    for step in range(steps):
        # Choose move type
        if rng.random() < 0.5:
            # Pivot move
            # Choose pivot index i (1 to n-2)
            i = rng.randint(1, n - 2)
            
            # The segment c[i:] is rotated.
            # We need to find a new position for the end of the segment such that
            # the rest of the segment fits without collision.
            # This is expensive to check fully. 
            # A faster approximation: Try moving the end of the segment to a neighbor
            # and see if the segment fits. If not, reject.
            
            # To make it efficient, we only try to move the *end* of the segment to a 
            # valid spot, assuming the rest of the segment rotates rigidly.
            # Actually, a pivot move rotates the entire segment around c[i-1].
            # The new position of c[j] for j >= i depends on the rotation.
            # In 3D, we can try 6 possible directions for the first step of the segment (c[i])
            # relative to c[i-1], excluding the previous direction.
            
            prev_pos = c[i-1]
            curr_pos = c[i]
            
            # Direction from prev to curr
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            dz = curr_pos[2] - prev_pos[2]
            
            # Possible new directions for c[i] relative to prev_pos
            # Exclude the reverse direction (back to prev_pos) and the current direction?
            # Actually, we exclude the position occupied by prev_pos.
            # We can try all 6 neighbors of prev_pos, except curr_pos itself?
            # No, we are moving the segment. c[i] moves to a new neighbor of prev_pos.
            
            new_dirs = []
            for np in _neighbours(prev_pos):
                if np != curr_pos: # Don't go back
                    new_dirs.append(np)
            
            if not new_dirs:
                continue
                
            new_start = rng.choice(new_dirs)
            
            # Now we need to check if the rest of the segment (c[i+1:]) fits
            # given that c[i] is at new_start.
            # This requires checking collisions for all j > i.
            # We can simulate the move partially.
            
            # Create a temporary occupied set without the segment c[i:]
            # But we need to know where c[i+1] goes. It's not a simple translation.
            # A pivot move is a rigid rotation.
            # In 3D lattice, a "pivot" is often defined as changing the direction of the bond (i-1, i)
            # and propagating that change? No, that's not rigid.
            
            # Standard Pivot Move:
            # Pick i. Rotate the segment i..n-1 around c[i-1].
            # The new position of c[j] is determined by the rotation.
            # To be efficient, we can just try to place c[i] at a new neighbor of c[i-1].
            # Then we need to find a valid position for c[i+1] that is a neighbor of new_c[i]
            # and not occupied by c[0..i-1] or new_c[i].
            # Then c[i+2] neighbor of new_c[i+1], etc.
            # This is a "reconnection" or "re-growth" of the tail, which is valid for lattice walks
            # if we allow the tail to change shape (non-rigid). This is often called "Tail Move" or "Regrow".
            # It preserves self-avoidance if we check each step.
            
            # Let's implement Tail Regrowth (Reconnect):
            # 1. Cut at i.
            # 2. Try to place c[i] at a neighbor of c[i-1] (not c[i-2] to avoid immediate backtracking? No, just not c[i-1] itself which is fixed, and not occupied).
            # 3. If valid, try to place c[i+1] at a neighbor of new_c[i] that is not occupied by 0..i-1 or new_c[i].
            # 4. Continue until end or failure.
            
            new_c = c[:i]
            new_occ = {p: idx for idx, p in enumerate(c[:i])}
            
            # Try to place the rest
            success = True
            prev_new_pos = c[i-1]
            
            for j in range(i, n):
                candidates = []
                for np in _neighbours(prev_new_pos):
                    if np not in new_occ:
                        # Avoid going back to the immediate predecessor in the new chain?
                        # If j > i, we shouldn't go to new_c[j-2]
                        if j > i and np == new_c[j-2]:
                            continue
                        candidates.append(np)
                
                if not candidates:
                    success = False
                    break
                
                # Choose randomly or with bias? Random is fine for Metropolis.
                chosen = rng.choice(candidates)
                new_c.append(chosen)
                new_occ[chosen] = j
                prev_new_pos = chosen
            
            if not success:
                continue
            
            # Calculate new energy
            new_contacts = _count_hh_contacts(sequence, new_c)
            
            # Metropolis acceptance
            delta = new_contacts - current_contacts
            if delta >= 0 or rng.random() < math.exp(delta / T):
                c = new_c
                occ = new_occ
                current_contacts = new_contacts
                
                if current_contacts > best_contacts:
                    best_contacts = current_contacts
                    best_chain = c[:]
        else:
            # Reversal Move (Flip a segment)
            # Pick i, j with i < j. Reverse c[i..j].
            # Check if the reversed segment is self-avoiding and doesn't collide with 0..i-1 or j+1..n-1.
            # The boundary connections (i-1, i) and (j, j+1) must remain valid unit steps.
            # Since it's a reversal, (i-1, i) becomes (i-1, j). Distance must be 1.
            # And (j, j+1) becomes (i, j+1). Distance must be 1.
            
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Check boundary distances
            if abs(c[i-1][0]-c[j][0]) + abs(c[i-1][1]-c[j][1]) + abs(c[i-1][2]-c[j][2]) != 1:
                continue
            if abs(c[j+1][0]-c[i][0]) + abs(c[j+1][1]-c[i][1]) + abs(c[j+1][2]-c[i][2]) != 1:
                continue
                
            # Check self-avoidance of the reversed segment against the rest
            # The segment c[i..j] is reversed.
            # We need to ensure that no two nodes in the reversed segment are the same (they aren't, it's a permutation)
            # and that no node in the reversed segment collides with nodes outside the segment.
            
            # Build set of outside nodes
            outside = set()
            for k in range(0, i):
                outside.add(c[k])
            for k in range(j+1, n):
                outside.add(c[k])
                
            # Check reversed segment
            valid = True
            for k in range(i, j+1):
                pos = c[j - (k - i)] # Reversed position
                if pos in outside:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Construct new chain
            new_c = c[:i] + c[i:j+1][::-1] + c[j+1:]
            
            new_contacts = _count_hh_contacts(sequence, new_c)
            
            delta = new_contacts - current_contacts
            if delta >= 0 or rng.random() < math.exp(delta / T):
                c = new_c
                occ = {p: idx for idx, p in enumerate(c)}
                current_contacts = new_contacts
                
                if current_contacts > best_contacts:
                    best_contacts = current_contacts
                    best_chain = c[:]
                    
        T = max(min_T, T - cooling)
        
    return best_chain

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Monte Carlo Chain Growth folding algorithm with contact-based bias and local search.
    """
    rng = random.Random(seed)
    n = len(sequence)
    
    start_pos = (0, 0, 0)
    
    best_chain = None
    best_energy = 0  # 0 is the energy of no contacts
    
    start_time = time.time()
    
    best_chain = None
    best_energy = 0  # Unfolded
    
    attempts = 0
    max_attempts = 5000
    
    # Track the top few chains for local search
    top_chains = []  # List of (energy, chain)
    
    while attempts < max_attempts:
        if time.time() - start_time > time_limit * 0.6:
            break
        
        # Vary parameters for diversity
        T = 0.2 + rng.random() * 0.5
        contact_weight = 2.0 + rng.random() * 2.0
        
        chain = _grow_contact_biased(sequence, rng, T=T, contact_weight=contact_weight)
        if chain is not None:
            # Calculate energy
            e = _calculate_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            
            # Keep top chains for local search
            top_chains.append((e, chain))
            top_chains.sort(key=lambda x: x[0])
            if len(top_chains) > 10:
                top_chains.pop()
        
        attempts += 1
    
    # Apply local search to the best chains if we have time
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 0.5 and best_chain is not None:
        # Apply local search to the top chains
        # Distribute remaining time among top chains
        num_ls = min(5, len(top_chains))
        if num_ls > 0:
            time_per_ls = remaining_time / num_ls
            for idx in range(num_ls):
                if time.time() - start_time > time_limit - 0.1:
                    break
                
                e, chain = top_chains[idx]
                # Scale steps with time
                steps = int(time_per_ls * 200) # Heuristic: 200 steps per second
                if steps < 50:
                    steps = 50
                    
                refined_chain = _local_search_contacts(sequence, chain, rng, steps=steps)
                refined_e = _calculate_energy(sequence, refined_chain)
                
                if refined_e < best_energy:
                    best_energy = refined_e
                    best_chain = refined_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with contact-based bias and local search.
    """
    n = len(sequence)
    # Base time 2s + 0.05s per residue
    time_limit = 2.0 + 0.05 * n
    if time_limit > 10.0:
        time_limit = 10.0
    if time_limit < 1.0:
        time_limit = 1.0

    # Use a unique seed for each call
    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    # Ensure format: list of lists
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
