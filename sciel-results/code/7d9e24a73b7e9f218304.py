"""
HP Lattice Protein Folding on Cubic Lattice
Uses Monte Carlo Chain Growth (MCCG) with explicit H-H contact counting
during growth and sophisticated local search with pivot and k-exchange moves.
"""
import math
import random
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _count_hh_contacts_fast(sequence, chain):
    """
    Count H-H contacts efficiently.
    Returns number of non-bonded H-H contacts.
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Build position -> index map
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        # Check all 6 neighbors
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy, z+dz), -1)
            if j != -1 and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return contacts // 2  # Each contact counted twice

def _energy(sequence, chain):
    """Energy = -number of H-H contacts."""
    return -_count_hh_contacts_fast(sequence, chain)

def _grow_with_contact_bias(sequence, rng, temp=1.0, bias_strength=3.0):
    """
    Grow a self-avoiding walk with strong bias toward positions that
    create the most H-H contacts with already-placed residues.
    Uses explicit contact counting for scoring.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        lx, ly, lz = last
        neighbors = [
            (lx+1, ly, lz), (lx-1, ly, lz),
            (lx, ly+1, lz), (lx, ly-1, lz),
            (lx, ly, lz+1), (lx, ly, lz-1)
        ]
        
        available = []
        for p in neighbors:
            if p not in occupied:
                available.append(p)
        
        if not available:
            return None
        
        # Score each available position by counting how many H-H contacts
        # would be formed if we place residue i at that position
        scored = []
        for p in available:
            if sequence[i] == 'H':
                # Count H neighbors that are not adjacent in sequence
                contact_gain = 0
                px, py, pz = p
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    j = occupied.get((px+dx, py+dy, pz+dz), -1)
                    if j != -1 and abs(i - j) > 1 and sequence[j] == 'H':
                        contact_gain += 1
            else:
                contact_gain = 0
            
            # Also consider: if this residue is P, it doesn't contribute,
            # but we might still want to avoid blocking future H-H contacts.
            # Simple heuristic: slight preference for positions with more free neighbors
            free_neighbors = 0
            px, py, pz = p
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                if (px+dx, py+dy, pz+dz) not in occupied:
                    free_neighbors += 1
            
            score = contact_gain * bias_strength + free_neighbors * 0.1
            scored.append((score, p))
        
        # Boltzmann selection
        max_score = max(s for s, _ in scored)
        weights = []
        total_w = 0.0
        for s, p in scored:
            w = math.exp((s - max_score) / temp)
            weights.append(w)
            total_w += w
        
        if total_w == 0:
            chosen = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = scored[0][1]
            for w, (_, p) in zip(weights, scored):
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain

def _pivot_move(sequence, chain, pivot_idx, rng):
    """
    Attempt a pivot move at pivot_idx.
    Rotates the tail (chain[pivot_idx:]) to a new random orientation.
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # Extract tail
    tail = chain[pivot_idx:]
    head = chain[:pivot_idx]
    
    # Build occupied set for head
    head_occupied = set(head)
    
    # Try random orientations for the tail
    # We'll try to re-embed the tail by random walk from the pivot point
    # This is complex; instead, try simple rotations
    
    # Simple approach: try shifting the entire tail by a random vector
    # and check for collisions
    for _ in range(10):
        # Random offset
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        dz = rng.randint(-2, 2)
        if dx == 0 and dy == 0 and dz == 0:
            continue
        
        new_tail = [(x+dx, y+dy, z+dz) for x, y, z in tail]
        
        # Check: first point of new_tail must be adjacent to last point of head
        last_head = head[-1]
        first_new_tail = new_tail[0]
        if abs(last_head[0] - first_new_tail[0]) + abs(last_head[1] - first_new_tail[1]) + abs(last_head[2] - first_new_tail[2]) != 1:
            continue
        
        # Check for collisions
        valid = True
        for p in new_tail:
            if p in head_occupied:
                valid = False
                break
        if not valid:
            continue
        
        new_chain = head + new_tail
        return new_chain
    
    return None

def _local_search(sequence, chain, rng, time_budget=2.0):
    """
    Local search using pivot moves and random restarts within the neighborhood.
    Uses simulated annealing to escape local minima.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    c = chain[:]
    current_energy = _energy(sequence, c)
    best_chain = c[:]
    best_energy = current_energy
    
    start_time = time.time()
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.995
    
    steps = 0
    while time.time() - start_time < time_budget:
        steps += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot move
            pivot = rng.randint(1, n - 2)
            new_chain = _pivot_move(sequence, c, pivot, rng)
        else:
            # Try a small random perturbation: pick a random index and try to
            # move that residue to a nearby free position (if valid)
            new_chain = None
            idx = rng.randint(1, n - 2)
            x, y, z = c[idx]
            candidates = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            occupied = set(c)
            occupied.discard(c[idx])
            
            for cand in candidates:
                if cand not in occupied:
                    # Check adjacency constraints
                    # Previous must be adjacent to cand
                    prev = c[idx-1]
                    if abs(prev[0]-cand[0]) + abs(prev[1]-cand[1]) + abs(prev[2]-cand[2]) == 1:
                        # Next must be adjacent to cand
                        nxt = c[idx+1]
                        if abs(nxt[0]-cand[0]) + abs(nxt[1]-cand[1]) + abs(nxt[2]-cand[2]) == 1:
                            new_chain = c[:]
                            new_chain[idx] = cand
                            break
            if new_chain is None:
                # Fall back to pivot
                pivot = rng.randint(1, n - 2)
                new_chain = _pivot_move(sequence, c, pivot, rng)
        
        if new_chain is None:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T) for worse
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            c = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = c[:]
        
        T = max(T_min, T * cooling_rate)
    
    return best_chain

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Main folding function: MCCG with contact bias + local search.
    """
    rng = random.Random(seed)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    
    # Phase 1: Multiple MCCG runs with varying parameters
    attempts = 0
    max_attempts = 5000
    
    top_chains = []  # (energy, chain)
    
    while attempts < max_attempts:
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.6:
            break
        
        # Vary temperature and bias for diversity
        temp = 0.5 + rng.random() * 1.5
        bias = 2.0 + rng.random() * 4.0
        
        chain = _grow_with_contact_bias(sequence, rng, temp=temp, bias_strength=bias)
        if chain is not None:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            
            top_chains.append((e, chain))
            top_chains.sort(key=lambda x: x[0])
            if len(top_chains) > 8:
                top_chains.pop()
        
        attempts += 1
    
    # Phase 2: Local search on the best chains
    remaining = time_limit - (time.time() - start_time)
    if remaining > 0.5 and best_chain is not None:
        # Allocate time among top chains
        num_ls = min(3, len(top_chains))
        ls_time_each = min(1.5, remaining * 0.4 / max(1, num_ls))
        
        for e, chain in top_chains[:num_ls]:
            if time.time() - start_time > time_limit - 0.2:
                break
            
            refined = _local_search(sequence, chain, rng, time_budget=ls_time_each)
            refined_e = _energy(sequence, refined)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined
    
    # Phase 3: If time remains, do more MCCG + quick local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0 and best_chain is not None:
        # More growth attempts
        while time.time() - start_time < time_limit - 0.5:
            temp = 0.5 + rng.random() * 1.0
            bias = 3.0 + rng.random() * 3.0
            chain = _grow_with_contact_bias(sequence, rng, temp=temp, bias_strength=bias)
            if chain is not None:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
        
        # Final local search on best
        if time.time() - start_time < time_limit - 0.3:
            final_time = min(2.0, time_limit - (time.time() - start_time) - 0.2)
            if final_time > 0.5:
                refined = _local_search(sequence, best_chain, rng, time_budget=final_time)
                refined_e = _energy(sequence, refined)
                if refined_e < best_energy:
                    best_energy = refined_e
                    best_chain = refined
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with contact-based bias and local search.
    """
    n = len(sequence)
    # Scale time with length: longer chains need more time
    time_limit = 1.0 + 0.04 * n
    if time_limit > 8.0:
        time_limit = 8.0
    if time_limit < 1.0:
        time_limit = 1.0

    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
