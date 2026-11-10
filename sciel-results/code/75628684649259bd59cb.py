# EVOLVE-BLOCK-START
import time
import random

_START_TIME = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill

def _neighbors(p):
    """Yield 6 neighbors of point p in 3D cubic lattice."""
    x, y, z = p
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _compute_energy(sequence, chain, occupied):
    """Compute energy: -number of H-H contacts (non-consecutive, adjacent in lattice)."""
    contacts = 0
    for i, pos in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for nb in _neighbors(pos):
            j = occupied.get(nb)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, chain, occupied):
    """Grow chain greedily from current state. Returns True on success."""
    for i in range(len(chain), len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbors(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for nb in _neighbors(site):
                    j = occupied.get(nb)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for nb in _neighbors(site):
                if nb not in occupied:
                    free += 1
            score = gain * 10 + free * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return False
        options.sort(reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    return True

def _local_search(sequence, chain, occupied, max_iters, rng, time_limit):
    """Local search: try single-residue moves to improve energy."""
    n = len(sequence)
    current_energy = _compute_energy(sequence, chain, occupied)
    best_energy = current_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    
    iters = 0
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        # Pick a random residue (not endpoints) to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try moving this residue to each neighboring site
        for new_pos in _neighbors(old_pos):
            if time.time() > time_limit:
                break
            
            # new_pos must not be occupied (except by adjacent chain residues)
            if new_pos in occupied:
                j = occupied[new_pos]
                if j != i - 1 and j != i + 1:
                    continue
            
            # Temporarily remove old_pos
            del occupied[old_pos]
            
            # Check no other non-adjacent chain residue is at new_pos
            conflict = False
            if new_pos in occupied:
                j = occupied[new_pos]
                if j != i - 1 and j != i + 1:
                    conflict = True
            
            if conflict:
                occupied[old_pos] = i
                continue
            
            # Place at new position
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _compute_energy(sequence, chain, occupied)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
            else:
                # Revert
                del occupied[new_pos]
                occupied[old_pos] = i
                chain[i] = old_pos
    
    return best_chain, best_occupied, best_energy

def _fold_sequence(sequence, dimension, seed):
    """Fold a single sequence with multiple restarts and local search."""
    rng = random.Random(seed)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0  # energy is negative, so 0 is worst (unfolded)
    
    # Determine number of restarts based on sequence length
    if n <= 48:
        num_restarts = 40
    elif n <= 64:
        num_restarts = 25
    elif n <= 100:
        num_restarts = 12
    else:
        num_restarts = 6
    
    # Local search iterations per restart
    if n <= 48:
        ls_iters = 800
    elif n <= 64:
        ls_iters = 400
    elif n <= 100:
        ls_iters = 200
    else:
        ls_iters = 80
    
    for restart in range(num_restarts):
        if time.time() > _DEADLINE:
            break
        
        # Start from origin
        start_pos = (0, 0, 0)
        chain = [start_pos]
        occupied = {start_pos: 0}
        
        # Greedy growth
        if not _greedy_grow(sequence, dimension, rng, chain, occupied):
            continue
        
        if len(chain) != n:
            continue
        
        # Local search
        chain, occupied, energy = _local_search(sequence, chain, occupied, ls_iters, rng, _DEADLINE)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain

def fold(sequence, dimension):
    """Fold the HP sequence on the 3D cubic lattice."""
    n = len(sequence)
    
    # Use a deterministic seed based on sequence
    seed = hash(sequence) % (2**31)
    
    chain = _fold_sequence(sequence, dimension, seed)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
