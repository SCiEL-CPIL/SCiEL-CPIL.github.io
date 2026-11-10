"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
# We have 26 sequences. 330 / 26 ~ 12.7s per sequence.
# We use a safety margin to ensure we don't timeout.
_TOTAL_START = time.time()
_TOTAL_BUDGET = 300.0  # Safety margin: 300s instead of 330s
_PER_SEQ_BUDGET = _TOTAL_BUDGET / 26.0 * 0.9  # 90% of average time per seq, ~11.4s * 0.9 = ~10.2s

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    Prefers sites with new H-H contacts, then free space, then random noise.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain: new H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Calculate free space: number of unoccupied neighbors
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: gain is primary, free space is secondary, noise for diversity
            # Using a smaller weight for free space to prioritize contacts
            score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score, site = max(options)
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """
    Calculate the energy of the chain.
    Energy = - (number of H-H contacts on non-adjacent lattice neighbors).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, deadline):
    """
    Hill-climbing local search using single-residue moves.
    A move is valid if residue i is moved to a site s such that:
    1. s is a neighbor of chain[i-1] and chain[i+1]
    2. s is not occupied by any other residue
    3. s != chain[i]
    """
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    # Precompute is_h for faster access
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of residues to avoid bias
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() >= deadline:
                break
                
            old_pos = best[i]
            prev_pos = best[i - 1]
            next_pos = best[i + 1]
            
            # Find candidate sites: neighbors of prev_pos AND next_pos
            # that are not occupied by anyone else
            candidates = []
            for site in _neighbours(prev_pos):
                if site == old_pos:
                    continue
                # Check if site is neighbor of next_pos
                # Manhattan distance must be 1
                if sum(abs(site[d] - next_pos[d]) for d in range(3)) == 1:
                    # Check if site is free
                    # It can only be occupied by i (which is old_pos, so site != old_pos)
                    # So if site is in occupied, it's occupied by someone else
                    if site not in occupied:
                        candidates.append(site)
            
            if not candidates:
                continue
            
            # Try moving to each candidate
            for site in candidates:
                # Calculate energy delta
                # Remove old_pos, add site
                # Delta = (New Contacts) - (Old Contacts)
                # Energy = -Contacts, so Delta_Energy = -Delta_Contacts
                # We want Delta_Energy < 0, so Delta_Contacts > 0
                
                delta_contacts = 0
                
                # If residue i is H, it contributes to contacts
                if is_h[i]:
                    # Lost contacts: neighbors of old_pos that are H and occupied by j != i
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts -= 1
                    
                    # Gained contacts: neighbors of site that are H and occupied by j != i
                    # Note: site is not in occupied yet, so we just check neighbors
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts += 1
                
                # If delta_contacts > 0, energy decreases (improves)
                if delta_contacts > 0:
                    # Apply move
                    del occupied[old_pos]
                    occupied[site] = i
                    best[i] = site
                    best_e -= delta_contacts  # Energy becomes more negative
                    improved = True
                    break  # Move to next i
            
            # If we improved, we can continue to next i or restart.
            # Continuing is faster but might miss global improvements.
            # Given time constraints, continuing is a good trade-off.
    
    return best, best_e

def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Many greedy growths
    2. Local search on the best candidates
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Time budget for this sequence
    # Ensure we don't exceed the global deadline
    global_start = _TOTAL_START
    current_time = time.time()
    elapsed_global = current_time - global_start
    remaining_global = _TOTAL_BUDGET - elapsed_global
    
    # Allocate time for this sequence, but not more than the global remaining time
    # and not more than the per-sequence budget
    seq_deadline = min(current_time + _PER_SEQ_BUDGET, global_start + _TOTAL_BUDGET)
    
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded)
    
    # Phase 1: Greedy growths
    # Use a fixed number of restarts based on length, but stop if time runs out
    # Shorter chains can afford more restarts
    if n <= 48:
        n_restarts = 500
    elif n <= 80:
        n_restarts = 300
    else:
        n_restarts = 150
    
    for restart in range(n_restarts):
        if time.time() >= seq_deadline - 2.0:  # Leave 2s for local search
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search
    if time.time() < seq_deadline - 0.5:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, seq_deadline - 0.5)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
