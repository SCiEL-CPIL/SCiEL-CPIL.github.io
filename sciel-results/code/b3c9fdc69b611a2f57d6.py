"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

# Precomputed neighbor offsets for 3D
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _grow(sequence, dimension, rng):
    """
    Optimized randomized greedy chain growth.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Precompute H positions for faster check? 
    # Actually, checking sequence[i] is fast.
    
    for i in range(1, n):
        last = chain[-1]
        # Get neighbors of last
        # In 3D, last is (x, y, z)
        lx, ly, lz = last
        
        # Candidate sites
        candidates = [
            (lx+1, ly, lz), (lx-1, ly, lz),
            (lx, ly+1, lz), (lx, ly-1, lz),
            (lx, ly, lz+1), (lx, ly, lz-1)
        ]
        
        best_score = -1
        best_site = None
        is_h = (sequence[i] == 'H')
        
        for site in candidates:
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts
            gain = 0
            if is_h:
                sx, sy, sz = site
                # Check 6 neighbors of site
                for dx, dy, dz in _OFFSETS:
                    neighbor = (sx+dx, sy+dy, sz+dz)
                    idx = occupied.get(neighbor)
                    if idx is not None:
                        if idx < i - 1: # Not adjacent in chain
                            if sequence[idx] == 'H':
                                gain += 1
            
            # Calculate free space (heuristic to avoid trapping)
            # Count empty neighbors
            free = 0
            sx, sy, sz = site
            for dx, dy, dz in _OFFSETS:
                neighbor = (sx+dx, sy+dy, sz+dz)
                if neighbor not in occupied:
                    free += 1
            
            # Score: gain is most important, free space is secondary to avoid dead ends
            # Add small random noise to break ties and explore
            score = gain + 0.1 * free + rng.random()
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            return None # Trapped
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _energy(sequence, chain):
    """
    Compute energy: -1 * (number of H-H contacts).
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            x, y, z = chain[i]
            # Check 6 neighbors
            for dx, dy, dz in _OFFSETS:
                neighbor = (x+dx, y+dy, z+dz)
                j = where.get(neighbor)
                if j is not None:
                    if j > i + 1: # Not adjacent
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts

def fold(sequence, dimension):
    """
    Best conformation of many randomized greedy growths.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative, so we want the most negative (lowest) value. 
                    # But initial 0 is "no contacts". 
                    # Wait, if no H-H contacts, energy is 0. 
                    # If we find contacts, energy becomes negative. 
                    # So we want to minimize energy (make it more negative).
                    # So initialize best_energy to 0 (or a large positive number if we want to force finding *some* structure, 
                    # but 0 is a valid valid state).
                    # Actually, if the sequence has no H, energy is 0. 
                    # If it has H, we want negative.
                    # So `if energy < best_energy` works if best_energy starts at 0.
                    # BUT, if the first chain has 0 contacts, best_energy stays 0.
                    # If the second has 1 contact, energy is -1, which is < 0, so it updates.
                    # This is correct.

    n = len(sequence)
    
    # Determine number of restarts based on time and chain length
    # Greedy growth is O(N^2) roughly due to set lookups and neighbor checks.
    # For N=136, it's fast. 
    # Let's try a high number of restarts, limited by deadline.
    
    start_time = time.time()
    
    # We can do many restarts. Let's try 1000.
    # If it's too slow, the deadline guard will stop it.
    max_restarts = 2000
    
    for _ in range(max_restarts):
        if _is_deadline_exceeded():
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Optional: Early stopping if we hit a very good score?
        # Hard to know what is "very good" without knowing the best known.
        # Just keep going.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
