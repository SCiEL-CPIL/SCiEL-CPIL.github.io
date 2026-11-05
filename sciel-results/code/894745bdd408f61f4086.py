"""
HP Lattice Protein Folding using Fast Incremental Monte Carlo Pivot Search
Strategy:
1. Multi-Start: Generate multiple initial conformations using fast greedy growth.
2. Fast Pivot Moves: Rotate the tail of the chain around a pivot point.
3. Incremental State Update: Update occupied set and chain in place, with careful revert on rejection.
4. Time-Adaptive: Allocate time dynamically, restarting if no improvement is found for a while.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.95  # Safety margin for 330s total budget

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    x, y, z = point
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _compute_energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    
    pos_map = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_map.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:
                        contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using randomized greedy growth.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 5.0 + free * 0.5 + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        
        if len(options) > 1 and rng.random() < 0.2:
            site = options[rng.randint(0, min(2, len(options)-1))][1]
        else:
            site = options[0][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _mc_pivot_fold(sequence, dimension, rng, time_limit):
    """
    Monte Carlo Pivot Search with Incremental State Updates.
    """
    n = len(sequence)
    if n <= 2:
        return _greedy_grow(sequence, dimension, rng)
    
    # Initialize with best of a few greedy starts
    best_chain = None
    best_energy = 0
    for _ in range(5):
        c = _greedy_grow(sequence, dimension, rng)
        if c is None:
            continue
        e = _compute_energy(sequence, c)
        if best_chain is None or e < best_energy:
            best_chain = c
            best_energy = e
    
    if best_chain is None:
        return [(i, 0, 0) for i in range(n)]
    
    chain = list(best_chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    start_time = time.time()
    last_improve_time = start_time
    
    # Precompute unit vectors
    unit_vecs = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
    ]
    
    while not _is_deadline_exceeded():
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Perform a batch of moves
        moves_per_batch = 2000 if n < 50 else 500
        
        for _ in range(moves_per_batch):
            if _is_deadline_exceeded():
                break
                
            # Pick pivot index i (1 to n-2)
            i = rng.randint(1, n - 2)
            pivot = chain[i]
            
            # Current vector from i to i+1
            v_old = (chain[i+1][0] - pivot[0], chain[i+1][1] - pivot[1], chain[i+1][2] - pivot[2])
            
            # Generate candidate new vectors
            # Exclude v_old
            candidates = [v for v in unit_vecs if v != v_old]
            
            if not candidates:
                continue
                
            v_new = candidates[rng.randint(0, len(candidates) - 1)]
            
            # New position for i+1
            q_new = (pivot[0] + v_new[0], pivot[1] + v_new[1], pivot[2] + v_new[2])
            
            # Check if q_new is occupied by head (0..i)
            if q_new in occupied:
                continue
                
            # Translation vector for the rest of the tail (i+2 .. n-1)
            dx = v_new[0] - v_old[0]
            dy = v_new[1] - v_old[1]
            dz = v_new[2] - v_old[2]
            
            # Check collisions for the rest of the tail
            valid = True
            for k in range(i + 2, n):
                p_old = chain[k]
                p_new = (p_old[0] + dx, p_old[1] + dy, p_old[2] + dz)
                if p_new in occupied:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Move is valid. Store old tail for potential revert.
            old_tail = chain[i+1:]
            
            # Apply move
            chain[i+1] = q_new
            for k in range(i + 2, n):
                p_old = chain[k]
                p_new = (p_old[0] + dx, p_old[1] + dy, p_old[2] + dz)
                chain[k] = p_new
            
            # Update occupied
            # Remove old tail
            for p in old_tail:
                del occupied[p]
            # Add new tail
            for k in range(i + 1, n):
                occupied[chain[k]] = k
            
            # Recompute energy
            new_energy = _compute_energy(sequence, chain)
            
            # Acceptance criterion
            T = 1.0 + 2.0 * math.exp(-0.05 * elapsed)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    last_improve_time = time.time()
            else:
                # Revert
                # Restore chain
                for k in range(i + 1, n):
                    chain[k] = old_tail[k - (i + 1)]
                
                # Revert occupied
                for k in range(i + 1, n):
                    del occupied[chain[k]]
                for p, idx in zip(old_tail, range(i + 1, n)):
                    occupied[p] = idx

        # Time check and restart
        if time.time() - last_improve_time > 2.0:
            new_chain = _greedy_grow(sequence, dimension, rng)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    chain = list(best_chain)
                    current_energy = best_energy
                    occupied = {p: i for i, p in enumerate(chain)}
                    last_improve_time = time.time()

    return best_chain

def fold(sequence, dimension):
    """Fold a protein sequence on the simple cubic lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    if n < 50:
        time_limit = 3.0
    elif n < 80:
        time_limit = 6.0
    else:
        time_limit = 10.0
        
    current_time = time.time()
    deadline = _get_deadline()
    remaining = deadline - current_time
    
    if remaining < 1.0:
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]
    
    if time_limit > remaining * 0.9:
        time_limit = remaining * 0.9
        
    best_chain = _mc_pivot_fold(sequence, dimension, rng, time_limit)
    
    pos_set = set(best_chain)
    if len(pos_set) != len(best_chain):
        greedy_chain = _greedy_grow(sequence, dimension, rng)
        if greedy_chain and len(set(greedy_chain)) == len(greedy_chain):
            best_chain = greedy_chain
        else:
            best_chain = [(i, 0, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
