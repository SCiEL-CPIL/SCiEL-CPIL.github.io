"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Deadline guard: 330s wall-clock kill, use 320s as safety margin
_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temp=0.0):
    """
    One chain growth. Uses a scoring function that balances:
    - New H-H contacts (primary)
    - Free space (to avoid trapping)
    - Random noise (for diversity)
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
            # Count new H-H contacts this site would make
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours (to avoid dead ends)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: contacts dominate, free space breaks ties, noise for diversity
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best (with some randomness if temp > 0)
        if temp > 0 and rng.random() < 0.3:
            # Sometimes pick randomly from top options
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _compute_contact_set(sequence, chain):
    """Return set of (i, j) pairs where i < j, both H, and chain[i] adjacent to chain[j]."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts


def _local_search(sequence, chain, dimension, rng, max_moves=500):
    """
    Local search with 2-opt reversals and incremental energy updates.
    Accepts any improvement (not just global best) to escape local minima.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_contacts = _compute_contact_set(sequence, best_chain)
    best_energy = -len(best_contacts)
    
    for move in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        # 2-opt: pick a < b, reverse chain[a:b+1]
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        
        # Ensure a and b are not adjacent (would just flip direction)
        if b - a == 1:
            continue
        
        # Try reversing segment [a, b]
        new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
        
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            continue
        
        # Check unit steps at the two junctions
        # Junction 1: between a-1 and a (if a > 0)
        if a > 0:
            diff = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
            if diff != 1:
                valid = False
        # Junction 2: between b and b+1 (if b < n-1)
        if valid and b < n - 1:
            diff = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
        
        if not valid:
            continue
        
        # Compute new energy
        new_contacts = _compute_contact_set(sequence, new_chain)
        new_energy = -len(new_contacts)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            best_contacts = new_contacts
    
    return best_chain, best_energy


def _polish(sequence, chain, dimension, rng, time_budget=5.0):
    """
    Final high-precision polish pass: try single-residue moves and short segment
    reversals with tighter acceptance criteria.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_contacts = _compute_contact_set(sequence, best_chain)
    best_energy = -len(best_contacts)
    
    start = time.time()
    moves_made = 0
    
    while time.time() - start < time_budget and moves_made < 500:
        if _get_remaining_time() < 2.0:
            break
        
        # Try a few different move types
        move_type = rng.choice(['single', 'short_rev', 'medium_rev'])
        
        if move_type == 'single':
            # Try moving a single residue to an adjacent free site
            # This requires the new site to be adjacent to both neighbors
            i = rng.randint(1, n - 2)
            current = best_chain[i]
            for site in _neighbours(current):
                if site == best_chain[i-1] or site == best_chain[i+1]:
                    continue
                # Check if site is free
                if site in best_chain:
                    continue
                # Check connectivity: i-1 must be adjacent to site, site adjacent to i+1
                d1 = sum(abs(best_chain[i-1][d] - site[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - best_chain[i+1][d]) for d in range(dimension))
                if d1 != 1 or d2 != 1:
                    continue
                
                new_chain = best_chain[:i] + [site] + best_chain[i+1:]
                new_contacts = _compute_contact_set(sequence, new_chain)
                new_energy = -len(new_contacts)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    best_contacts = new_contacts
                    moves_made += 1
                    break
        
        elif move_type == 'short_rev':
            # Reverse a short segment (length 2-4)
            seg_len = rng.randint(2, 4)
            a = rng.randint(0, n - 1 - seg_len)
            b = a + seg_len
            if b >= n:
                continue
            new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
            
            # Quick validity check
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            if a > 0:
                diff = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if valid and b < n - 1:
                diff = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if not valid:
                continue
            
            new_contacts = _compute_contact_set(sequence, new_chain)
            new_energy = -len(new_contacts)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                best_contacts = new_contacts
                moves_made += 1
        
        elif move_type == 'medium_rev':
            # Reverse a medium segment (length 5-10)
            seg_len = rng.randint(5, 10)
            a = rng.randint(0, n - 1 - seg_len)
            b = a + seg_len
            if b >= n:
                continue
            new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
            
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            if a > 0:
                diff = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if valid and b < n - 1:
                diff = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if not valid:
                continue
            
            new_contacts = _compute_contact_set(sequence, new_chain)
            new_energy = -len(new_contacts)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                best_contacts = new_contacts
                moves_made += 1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Many greedy growths with varying temperatures
    2. Local search (2-opt) on the best candidates
    3. Final polish pass
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # start with 0 (no contacts)
    
    # Phase 1: Greedy growths
    # Use more restarts for shorter chains, fewer for longer
    if n <= 48:
        n_restarts = 300
    elif n <= 80:
        n_restarts = 150
    else:
        n_restarts = 80
    
    for restart in range(n_restarts):
        if _get_remaining_time() < 5.0:
            break
        
        # Vary temperature for diversity
        temp = 0.0 if restart < n_restarts // 2 else rng.uniform(0.1, 1.0)
        
        chain = _greedy_grow(sequence, dimension, rng, temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 2: Local search on best chain
    if best_chain is not None and _get_remaining_time() > 10.0:
        # Do local search with time budget
        max_moves = 3000
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, max_moves)
    
    # Phase 3: Final polish pass
    if best_chain is not None and _get_remaining_time() > 5.0:
        time_budget = min(5.0, _get_remaining_time() - 3.0)
        best_chain, best_energy = _polish(sequence, best_chain, dimension, rng, time_budget)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
