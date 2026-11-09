"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2


def _grow_csp(sequence, dimension, rng, bias=0.5):
    """
    Constraint-Satisfaction-Problem (CSP) framed greedy growth.
    Prioritizes moves that maximize immediate H-H contacts while maintaining 
    sufficient free space to avoid dead-ends.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0) for i in range(n)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # Pre-compute H positions for quick lookup? No, sequence is static.
    
    for i in range(1, n):
        candidates = []
        x, y = chain[-1]
        
        for dx, dy in dirs:
            new_pt = (x + dx, y + dy)
            if new_pt in occupied:
                continue
            
            # 1. Contact Gain
            gain = 0
            if sequence[i] == 'H':
                for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                               (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                    q = (nx, ny)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # 2. Free Space (Degree of freedom)
            free = 0
            for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                           (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                if (nx, ny) not in occupied:
                    free += 1
            
            # 3. Avoidance of "traps" (optional heuristic: penalize if free==1 and i < n-2)
            penalty = 0
            if free == 1 and i < n - 2:
                penalty = 2.0 # Strong penalty for creating a dead-end early
            
            # Score: Weighted sum of gain and free space, minus penalty
            # Bias controls trade-off: high bias -> prefer contacts, low bias -> prefer space
            score = (gain * 10.0) + (free * bias) - penalty
            
            candidates.append((score, new_pt, gain, free))
        
        if not candidates:
            return None
        
        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Stochastic choice: pick from top few to allow diversity
        # If top score is significantly better, pick it. Otherwise, randomize among top 3.
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 1.0:
            # Ambiguous, pick randomly from top 3
            top_k = min(3, len(candidates))
            choice = rng.choice(candidates[:top_k])
        else:
            choice = candidates[0]
            
        new_pt = choice[1]
        occupied[new_pt] = i
        chain.append(new_pt)
        
    return chain


def _local_move(chain, idx, direction, occupied, sequence, energy):
    """
    Attempt to move residue at idx to an adjacent empty spot.
    Returns (success, new_chain, new_energy, new_occupied).
    Uses incremental energy update.
    """
    n = len(chain)
    if idx <= 0 or idx >= n - 1:
        return False, None, None, None
    
    old_pt = chain[idx]
    new_pt = (old_pt[0] + direction[0], old_pt[1] + direction[1])
    
    # Check if new_pt is occupied by a non-neighbor
    if new_pt in occupied:
        occupant = occupied[new_pt]
        if abs(occupant - idx) > 1:
            return False, None, None, None
        # If occupant is a neighbor, we can't move there (would break chain connectivity or overlap)
        # Actually, if occupant is idx-1 or idx+1, it's a bond. Moving into a bond is invalid.
        return False, None, None, None
    
    # Check if new_pt is within bounds? No, lattice is infinite.
    
    # Check self-avoidance with the rest of the chain
    # We already checked `new_pt in occupied`. 
    # But we must ensure we don't create a collision with the immediate neighbors if they shift? 
    # No, local move only moves one residue. The neighbors stay put.
    # So we just need to ensure new_pt is not occupied by anyone other than idx (which it isn't, since we removed it) 
    # and not occupied by neighbors (checked above).
    
    # Wait, `occupied` contains all residues. 
    # If `new_pt` is in `occupied`, it must be a neighbor (idx-1 or idx+1) for the move to be geometrically possible 
    # without crossing? No, if it's occupied by idx-1, we are trying to move onto idx-1. That's invalid.
    # So if `new_pt in occupied`, return False.
    
    # Let's re-verify: 
    # 1. new_pt must be empty.
    if new_pt in occupied:
        return False, None, None, None

    # 2. new_pt must not be adjacent to the neighbors in a way that creates a bond?
    # The chain is ... (idx-1) - idx - (idx+1) ...
    # If we move idx to new_pt, the bonds become (idx-1)-new_pt and new_pt-(idx+1).
    # These must be unit steps. Since new_pt is adjacent to old_pt, and old_pt was adjacent to neighbors,
    # new_pt is NOT necessarily adjacent to the neighbors.
    # Example:
    # (0,0) - (1,0) - (2,0)
    # Move idx=1 (1,0) to (1,1).
    # New bonds: (0,0)-(1,1) [dist sqrt(2), INVALID]
    # So we must check that new_pt is adjacent to chain[idx-1] and chain[idx+1].
    
    prev_pt = chain[idx-1]
    next_pt = chain[idx+1]
    
    if abs(new_pt[0] - prev_pt[0]) + abs(new_pt[1] - prev_pt[1]) != 1:
        return False, None, None, None
    if abs(new_pt[0] - next_pt[0]) + abs(new_pt[1] - next_pt[1]) != 1:
        return False, None, None, None
        
    # 3. Check self-avoidance: new_pt must not be occupied by any other residue.
    # We already checked `new_pt in occupied`. 
    # But wait, `occupied` is the current state. 
    # If we remove old_pt and add new_pt, is new_pt free? 
    # Yes, if `new_pt not in occupied`.
    
    # Calculate energy change
    # Remove contacts involving old_pt
    # Add contacts involving new_pt
    
    # Contacts are H-H pairs at distance 1 (not bonded).
    # For a residue i, its potential contacts are neighbors j where |i-j|>1 and seq[j]=='H'.
    
    # Let's compute delta E.
    # E = -0.5 * sum_{i,j} H_i H_j [dist(i,j)==1] [ |i-j|>1 ]
    
    # Change in contacts for residue i (moving from old_pt to new_pt):
    # Old contacts: j such that dist(old_pt, chain[j]) == 1 and |i-j|>1 and seq[j]=='H'
    # New contacts: j such that dist(new_pt, chain[j]) == 1 and |i-j|>1 and seq[j]=='H'
    
    # Note: The contacts between i and its neighbors (i-1, i+1) are never counted as energy contacts 
    # because of the |i-j|>1 constraint. So moving i doesn't affect the "bond" status in terms of energy, 
    # but it does affect whether i is adjacent to other H's.
    
    # However, moving i might change whether i-1 is adjacent to i+1? No, i-1 and i+1 are fixed.
    # Moving i might change whether i-1 is adjacent to i-2? No.
    # The ONLY energy terms that change are those involving residue i.
    # Terms involving (i-1, i) and (i, i+1) are excluded from energy by definition.
    
    # So:
    # delta_contacts = (new_H_contacts - old_H_contacts)
    # new_E = E - delta_contacts
    
    # Wait, is it that simple?
    # Energy = - (number of H-H contacts).
    # If we gain 1 contact, energy decreases by 1.
    # If we lose 1 contact, energy increases by 1.
    
    # Let's count H-neighbors of old_pt (excluding i-1, i+1)
    old_h_count = 0
    if sequence[idx] == 'H':
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (old_pt[0]+dx, old_pt[1]+dy)
            j = occupied.get(neighbor)
            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                old_h_count += 1
    else:
        # If sequence[idx] is P, it doesn't form H-H contacts.
        # But does moving it affect contacts between OTHER residues?
        # No, P residues don't contribute to H-H contacts.
        # However, moving a P residue might "unblock" or "block" a contact between two H's?
        # No, contacts are defined by lattice positions. 
        # If i is P, moving it doesn't change the fact that two other H's are adjacent or not, 
        # UNLESS the move of i somehow allows two other H's to become adjacent? 
        # No, the other H's are fixed in position. Their adjacency is fixed.
        # So if sequence[idx] == 'P', delta_energy = 0.
        # Wait! Is that true?
        # Energy is sum over ALL pairs. 
        # If I move residue i (which is P), do I change the distance between any other pair (j,k)?
        # No. The positions of all other residues j != i remain unchanged.
        # Therefore, the set of pairs (j,k) with j,k != i remains unchanged.
        # The only pairs involving i are (i, j). Since i is P, these pairs contribute 0 to energy.
        # So yes, moving a P residue changes energy by 0.
        
        # BUT, we must still check self-avoidance.
        # If we move P, energy is unchanged.
        
        pass

    new_h_count = 0
    if sequence[idx] == 'H':
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (new_pt[0]+dx, new_pt[1]+dy)
            # Check if neighbor is occupied by an H
            # Note: We must check against the CURRENT occupied set, 
            # but we are conceptually moving i. 
            # The neighbors of new_pt are the same as neighbors of old_pt, except for the direction we moved?
            # No, we just check the 4 neighbors of new_pt in the CURRENT grid.
            # If a neighbor is i-1 or i+1, it's excluded by |j-i|>1.
            j = occupied.get(neighbor)
            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                new_h_count += 1
    
    # The change in energy is - (new_h_count - old_h_count)
    # Because E = -contacts.
    # delta_E = E_new - E_old = -(new_contacts - old_contacts)
    
    delta_contacts = new_h_count - old_h_count
    delta_energy = -delta_contacts
    
    new_energy = energy + delta_energy
    
    # Update chain and occupied
    new_chain = chain[:]
    new_chain[idx] = new_pt
    
    new_occupied = occupied.copy()
    del new_occupied[old_pt]
    new_occupied[new_pt] = idx
    
    return True, new_chain, new_energy, new_occupied


def _anneal_csp(sequence, initial_chain, budget_seconds, rng):
    """
    Simulated Annealing with Local Moves and Incremental Energy.
    """
    n = len(sequence)
    if n <= 2:
        return initial_chain
    
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # Build initial occupied map
    occupied = {}
    for i, p in enumerate(current_chain):
        occupied[p] = i
    
    T = 2.0
    T_min = 0.05
    cooling_rate = 0.9995
    
    start_t = time.time()
    steps = 0
    no_improve = 0
    max_no_improve = 1000
    
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    while _remaining_time() > budget_seconds:
        if no_improve > max_no_improve:
            break
        
        # Choose a random residue to move
        # Avoid ends
        idx = rng.randint(1, n - 2)
        
        # Choose a random direction
        dx, dy = rng.choice(dirs)
        
        success, new_chain, new_energy, new_occupied = _local_move(
            current_chain, idx, (dx, dy), occupied, sequence, current_energy
        )
        
        if not success:
            no_improve += 1
            # Small penalty for failed move? Or just count it.
            # To avoid getting stuck in a "no move possible" state, we might want to try another idx.
            # But for now, just increment no_improve.
            continue
        
        delta_E = new_energy - current_energy
        
        if delta_E <= 0 or rng.random() < exp(-delta_E / T):
            # Accept
            current_chain = new_chain
            current_energy = new_energy
            occupied = new_occupied
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = current_chain
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        steps += 1
        if steps % 50000 == 0:
            # Check time more frequently
            if time.time() - start_t > budget_seconds:
                break

    return best_chain


def fold(sequence, dimension):
    """Best conformation found via CSP growth + CSP Annealing."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    candidates = []
    
    # Phase 1: Generate diverse initial conformations
    # Use different biases for the CSP growth
    biases = [0.1, 0.5, 1.0, 2.0]
    num_attempts_per_bias = 5
    
    for bias in biases:
        for _ in range(num_attempts_per_bias):
            if _remaining_time() < 10.0:
                break
            chain = _grow_csp(sequence, dimension, rng, bias)
            if chain is not None:
                candidates.append(chain)
        
        if _remaining_time() < 10.0:
            break
    
    if not candidates:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Evaluate and sort
    candidates_with_energy = []
    for c in candidates:
        e = _energy(sequence, c)
        candidates_with_energy.append((e, c))
    
    candidates_with_energy.sort(key=lambda x: x[0])
    
    best_energy = candidates_with_energy[0][0]
    best_chain = candidates_with_energy[0][1]
    
    # Phase 2: Anneal the top candidates
    # Allocate time based on remaining budget
    # Top 3 candidates get the most time
    
    num_to_anneal = min(3, len(candidates_with_energy))
    
    # Time allocation: 50% for best, 30% for second, 20% for third (if available)
    # Or simply split remaining time evenly among top 3, with a minimum for each.
    
    remaining_time = _remaining_time()
    # Reserve 5s for final checks/return
    search_time = max(1.0, remaining_time - 5.0)
    
    if num_to_anneal == 1:
        budgets = [search_time]
    elif num_to_anneal == 2:
        budgets = [search_time * 0.6, search_time * 0.4]
    else:
        budgets = [search_time * 0.5, search_time * 0.3, search_time * 0.2]
    
    for i in range(num_to_anneal):
        if _remaining_time() < 5.0:
            break
            
        e, c = candidates_with_energy[i]
        budget = min(budgets[i], _remaining_time() - 3.0)
        if budget < 0.5:
            break
        
        annealed = _anneal_csp(sequence, c, budget, rng)
        e_annealed = _energy(sequence, annealed)
        
        if e_annealed < best_energy:
            best_energy = e_annealed
            best_chain = annealed
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
