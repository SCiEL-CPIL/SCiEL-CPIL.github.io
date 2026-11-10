"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 170.0

def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied:
                    free += 1
            # Weight gain heavily, then free space, then noise
            score = gain * 3.0 + 0.1 * free + rng.random()
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
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _calc_energy_delta(sequence, chain, occupied, i, new_pos):
    """
    Calculate the change in energy if residue i moves from chain[i] to new_pos.
    Energy = -contacts. So delta_energy = -delta_contacts.
    We return delta_energy.
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    # Check validity first? No, assume caller checks validity.
    # We need to count contacts involving i that are broken and formed.
    
    # A contact exists between residue i and j if:
    # 1. sequence[i] == 'H' and sequence[j] == 'H'
    # 2. j is a neighbor of i on the lattice
    # 3. abs(i - j) > 1 (not adjacent in chain)
    
    delta_contacts = 0
    
    # 1. Remove contacts involving i at old_pos
    if sequence[i] == 'H':
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                # This contact is broken
                delta_contacts -= 1
                
    # 2. Add contacts involving i at new_pos
    if sequence[i] == 'H':
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            # j must exist, not be i, not be adjacent in chain, and be H
            # Note: occupied contains the old_pos for i. We assume the move is valid 
            # so new_pos is not occupied by another residue.
            # However, occupied still has old_pos mapped to i.
            # If q == old_pos, j == i, but we skip j==i.
            # If q is occupied by another residue, j is that residue.
            if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                # This contact is formed
                # But wait, if q == old_pos, j is i (since occupied[old_pos]=i). 
                # We skip j==i.
                # What if the neighbor q is the same as old_pos? 
                # The move is from old_pos to new_pos. 
                # The neighbors of new_pos might include old_pos.
                # If q == old_pos, occupied[q] is i. We skip j==i. Correct.
                delta_contacts += 1
                
    # The delta in energy is the negative of the delta in contacts
    return -delta_contacts

def _local_optimize_sa(sequence, chain, occupied, time_budget=2.0):
    """
    Simulated Annealing optimization with incremental energy updates.
    Returns improved chain, occupied, and energy.
    """
    n = len(chain)
    if n <= 3:
        return chain, occupied, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    # Estimate steps based on time budget
    # Each step is O(1) energy update + O(1) move validation
    # Let's try a fixed number of steps per second, or just run until time limit.
    # Given 10 sequences and 170s total, we have ~17s per sequence.
    # But fold is called per sequence. The deadline is global.
    # We should use a portion of the remaining time.
    remaining_time = _DEADLINE - time.time()
    if remaining_time <= 0:
        return best_chain, best_occupied, best_energy
        
    # Use up to 5 seconds per sequence for SA, or 20% of remaining time, whichever is smaller.
    # Actually, let's just run SA for a fixed number of iterations that scales with n.
    # Or run until time slice is used.
    # Let's allocate a slice of time for this call.
    # If this is the first call, we have 170s. If last, we have less.
    # Simple strategy: run SA for min(5.0, remaining_time/10) seconds.
    sa_time_budget = min(5.0, remaining_time / 10.0)
    
    start_time = time.time()
    steps = 0
    # Number of attempts per second estimate: ~10000 simple moves/sec in Python?
    # Let's just loop until time runs out.
    
    while time.time() - start_time < sa_time_budget:
        # Pick a random residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Pick a random neighbor
        neighbors = _neighbours2d(old_pos)
        new_pos = neighbors[rng.randint(0, 3)]
        
        # Check validity
        # 1. new_pos not occupied by another residue
        if new_pos in occupied and occupied[new_pos] != i:
            continue
            
        # 2. Unit step check with i-1 and i+1
        if i > 0:
            px, py = chain[i-1]
            if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                continue
        if i < n - 1:
            px, py = chain[i+1]
            if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                continue
            
        # Calculate energy delta
        delta_e = _calc_energy_delta(sequence, chain, occupied, i, new_pos)
        
        # Determine acceptance
        accept = False
        if delta_e < 0:
            accept = True
        else:
            # Calculate temperature
            # Linear cooling
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / sa_time_budget)
            T = T_start * math.exp(-progress * 10.0) # Fast cooling
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True
            
        if accept:
            # Apply move
            occupied[old_pos] = i # Just in case, though we will update
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_occupied = dict(occupied)
                
        steps += 1
        
    return best_chain, best_occupied, best_energy

def fold(sequence, dimension):
    """Best conformation with deadline-aware search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # Time management
    start_time = time.time()
    
    # 1. Greedy Growth Phase
    # Run multiple growths to get diverse starts
    # Budget: 50% of available time for growths? Or fixed number?
    # Let's do 50 growths first.
    growths = []
    for _ in range(50):
        if time.time() > _DEADLINE:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            growths.append(chain)
            
    if not growths:
        # Fallback straight line
        return [list(p) for p in [(i, 0) for i in range(n)]]

    # 2. Refinement Phase
    # For each growth, run SA.
    # We can prioritize the best greedy results.
    # Sort growths by energy
    scored_growths = []
    for chain in growths:
        occupied = {p: i for i, p in enumerate(chain)}
        e = _energy(sequence, chain)
        scored_growths.append((e, chain, occupied))
        
    scored_growths.sort(key=lambda x: x[0])
    
    # Run SA on the top K starts
    # K depends on time remaining
    remaining = _DEADLINE - time.time()
    if remaining <= 0:
        # Return best greedy
        best_e, best_c, _ = scored_growths[0]
        return [list(p) for p in best_c]
    
    # Allocate time to SA runs
    # Let's run SA on top 5 starts, giving each 1/5 of the remaining time (capped).
    top_k = min(5, len(scored_growths))
    time_per_sa = min(3.0, remaining / top_k) # Cap 3s per SA run
    
    for i in range(top_k):
        if time.time() > _DEADLINE:
            break
        e, chain, occupied = scored_growths[i]
        # Run SA
        opt_chain, opt_occupied, opt_e = _local_optimize_sa(sequence, chain, occupied, time_budget=time_per_sa)
        
        if opt_e < best_energy:
            best_energy = opt_e
            best = opt_chain
            
        # Early stopping if we hit theoretical min? No, we don't know min.
        # If we have a very good energy, maybe stop?
        # For now, just finish the allocated time.

    if best is None:
        best = scored_growths[0][1]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
