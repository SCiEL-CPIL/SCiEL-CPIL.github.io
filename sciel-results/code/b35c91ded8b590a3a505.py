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


def _local_search(sequence, chain, dimension, rng, max_moves=500):
    """
    Simple local search: try flipping small segments or moving single residues.
    Uses 2-opt style: pick two non-adjacent positions and try to re-route.
    For simplicity, we do single-residue moves: pick a residue, try to move it
    to an adjacent free site, if the chain remains self-avoiding.
    """
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        # Pick a random interior residue to try moving
        i = rng.randint(1, len(best_chain) - 2)
        
        # Try all 6 neighbours of current position
        current = best_chain[i]
        moved = False
        for site in _neighbours(current):
            if site == best_chain[i - 1] or site == best_chain[i + 1]:
                continue
            if site in occupied:
                continue
            
            # Check that moving i to site doesn't create overlap
            # (we removed current, so check site is free)
            # Also check the chain is still valid: i-1 and i+1 must be adjacent to site? No,
            # we're moving residue i, so i-1 connects to site and site connects to i+1? 
            # Actually in a walk, moving a single residue requires that site is adjacent
            # to both i-1's position and i+1's position. That's very restrictive.
            
            # Better: try 2-opt reversal. Pick two positions a < b, reverse the segment.
            # This is more effective for lattice walks.
            break
        
        # 2-opt: pick a < b, reverse chain[a:b+1]
        a = rng.randint(0, len(best_chain) - 2)
        b = rng.randint(a + 1, len(best_chain) - 1)
        
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
        
        # Check unit steps
        for k in range(len(new_chain) - 1):
            diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            occupied = {p: i for i, p in enumerate(best_chain)}
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Many greedy growths with varying temperatures
    2. Local search (2-opt) on the best candidates
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
        max_moves = 2000
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, max_moves)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
