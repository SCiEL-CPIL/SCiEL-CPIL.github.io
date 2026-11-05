"""HP lattice protein folding using hybrid growth and Monte Carlo refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _grow(sequence, dimension, rng):
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
            # Heuristic: strongly prefer contacts, slightly prefer open space to avoid trapping
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _refine(sequence, chain, rng, time_limit):
    """Monte Carlo local search using kink moves (move a residue to an adjacent empty site)."""
    if not chain:
        return chain
    
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    # Build initial occupancy map
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    # Adaptive temperature schedule
    T_start = 1.0
    T_end = 0.05
    total_steps_target = 50000  # Rough estimate, will be cut by time limit
    
    # We don't know exact steps possible, so we loop until time runs out
    # But we need to be careful not to take too long in a single step.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Choose a random index to move. 
        # Moving ends is less likely to break connectivity but less flexible.
        # Moving internal residues (kinks) is more powerful.
        # Let's pick i in [1, n-2] with some probability, and ends with lower prob.
        # Actually, moving an end is just extending/retracting, which is not allowed in standard lattice walk moves 
        # unless we treat it as a "pivot" or "crankshaft". 
        # Standard "move" in SA for SAWs is often a "kink move" (move residue i to neighbor j)
        # where i is not 0 or n-1, and the new position is empty.
        
        # Let's stick to simple kink moves for internal residues to ensure self-avoidance is easy to check.
        if n > 2:
            i = rng.randint(1, n - 2)
        else:
            break
        
        old_pos = current_chain[i]
        
        # Find valid moves
        valid_moves = []
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            # Check if new_pos is adjacent to previous or next residue (would create 2-step jump or collision)
            # Actually, the move is valid if new_pos is not occupied.
            # But we must ensure the chain remains a valid walk.
            # Since we only move one residue to an adjacent empty spot, the bonds (i-1, i) and (i, i+1)
            # become (i-1, new_pos) and (new_pos, i+1).
            # We must ensure new_pos is not the same as old_pos (obvious)
            # And that new_pos is not adjacent to i-1 or i+1 in a way that creates a self-intersection?
            # No, self-avoidance is just that no two residues occupy the same site.
            # The only constraint is that new_pos is not occupied by any other residue.
            # However, we must ensure that the move doesn't cause the chain to intersect itself 
            # in a way that wasn't there before. Since we're moving i to an empty spot, 
            # and removing i from old_pos, the only new collisions could be with neighbors of i.
            # But neighbors of i are already in the chain. If new_pos is adjacent to i-1 or i+1, 
            # it's fine as long as it's not the *same* site.
            # Wait, if new_pos is adjacent to i-1, that's a valid bond.
            # The only invalid case is if new_pos is occupied.
            
            # One subtle issue: if new_pos is adjacent to i-1, and i-1 is adjacent to new_pos, 
            # is that a problem? No.
            # Is it a problem if new_pos == current_chain[i-1]? Yes, that's occupied.
            # So checking `new_pos in occupied` is sufficient for self-avoidance.
            
            valid_moves.append(new_pos)
        
        if not valid_moves:
            continue
        
        new_pos = rng.choice(valid_moves)
        
        # Calculate energy delta
        # Remove old_pos from occupied, add new_pos
        # Energy change depends on H-H contacts involving residue i and its neighbors
        # Residues involved: i-1, i, i+1
        
        # Let's just recompute energy locally for speed? 
        # Or just recompute full energy? For N=100, full energy is O(N^2) or O(N) with map.
        # O(N) with map is fast. Let's do a quick delta calculation.
        
        # Contacts lost:
        # 1. Between i and i-1? No, they are adjacent in chain, so not a contact.
        # 2. Between i and i+1? No.
        # 3. Between i and any other j? 
        #    We need to check neighbors of old_pos and new_pos.
        
        # Let's use a helper to compute energy contribution of a site given the occupied map and sequence.
        # But occupied map changes.
        
        # Simpler: 
        # 1. Temporarily update chain and occupied
        # 2. Compute delta energy
        # 3. If rejected, revert
        
        # Let's try a more efficient delta calculation.
        # Energy = - (number of H-H contacts)
        # Delta Energy = - (Delta Contacts)
        
        # Contacts involving residue i:
        # Old: Pairs (i, j) where j is neighbor of old_pos, j != i-1, j != i+1, and seq[j]=='H', seq[i]=='H'
        # New: Pairs (i, j) where j is neighbor of new_pos, j != i-1, j != i+1, and seq[j]=='H', seq[i]=='H'
        
        # Also, moving i might affect contacts between other residues? No, only i moves.
        
        delta_contacts = 0
        if sequence[i] == 'H':
            # Old neighbors of i that are H and not adjacent in chain
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta_contacts -= 1 # Losing this contact
            
            # New neighbors of i that are H and not adjacent in chain
            # Note: occupied still has old_pos for i. We need to be careful.
            # Let's assume we check against the *current* occupied map, but excluding i itself.
            # Since new_pos is not in occupied, we can just check neighbors of new_pos.
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    # But wait, if q is old_pos, it is in occupied (as i).
                    # We must exclude i itself.
                    if j == i:
                        continue
                    delta_contacts += 1 # Gaining this contact
        
        delta_energy = -delta_contacts
        
        # Metropolis acceptance
        if delta_energy < 0 or rng.random() < math.exp(-delta_energy / max(T_end, 1e-9)):
            # Accept
            current_chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i
            current_energy += delta_energy
        # Else reject, do nothing
    
    return current_chain

def fold(sequence, dimension):
    """Best conformation using hybrid growth and refinement."""
    start_global = time.time()
    total_time_limit = 110.0 # 120s budget, leave 10s for safety/overhead
    
    # Allocate time per sequence? The task says "fold is called once per benchmark sequence".
    # So this function is called 10 times.
    # The 120s budget is for ALL sequences.
    # We don't know how much time is left.
    # We should check global time if possible, but we don't have a global clock shared across calls 
    # unless we store it in a module variable.
    
    # Let's use a module-level timer.
    global _global_start_time
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
    
    elapsed_total = time.time() - fold._start_time
    remaining = total_time_limit - elapsed_total
    
    if remaining <= 0:
        # Fallback: return a straight line
        return [[i, 0] for i in range(len(sequence))]
    
    # Allocate a portion of remaining time to this sequence
    # Simple heuristic: assume 10 sequences, so 1/10 of remaining time?
    # Or just use a fixed fraction, e.g., 90% of remaining time / (10 - calls_made)?
    # We don't know calls_made.
    # Let's just use a safe fraction of the remaining time, e.g., 1/10 of remaining, 
    # but not less than 5 seconds.
    time_budget = max(5.0, remaining * 0.1)
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worst (unfolded)
    
    # Phase 1: Generate diverse initial structures
    num_growths = 20
    initial_chains = []
    for _ in range(num_growths):
        chain = _grow(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback
        return [[i, 0] for i in range(len(sequence))]
    
    # Phase 2: Refine each initial structure
    # Distribute time budget among refinements
    refine_time_each = time_budget / len(initial_chains)
    
    for chain in initial_chains:
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
        
        # Refine
        refined = _refine(sequence, chain, rng, refine_time_each)
        e_ref = _energy(sequence, refined)
        if e_ref < best_energy:
            best_energy = e_ref
            best_chain = refined
        
        # Check global time
        if time.time() - fold._start_time > total_time_limit - 5.0:
            break
            
    if best_chain is None:
        best_chain = initial_chains[0]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
