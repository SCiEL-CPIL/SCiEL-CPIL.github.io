"""HP lattice protein folding on the cubic lattice - fast greedy + local search."""
import time
import random
import math

# EVOLVE-BLOCK-START

_NEIGHBORS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _get_deadline():
    """Return the global deadline (330s wall clock with safety margin)."""
    if not hasattr(_get_deadline, '_start'):
        _get_deadline._start = time.time()
    return _get_deadline._start + 325.0

def _time_left():
    """Return seconds remaining before deadline."""
    return max(0.0, _get_deadline() - time.time())

def _neighbors(point):
    """Yield the 6 neighbors of a point on the cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _grow(sequence, rng):
    """
    Greedy chain growth: at each step, pick a free neighbor that maximizes
    new H-H contacts, with random tie-breaking.
    Returns (chain, occupied_dict) or None if growth fails.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)], {(0, 0, 0): 0}
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        best_score = -1
        best_sites = []
        
        for site in _neighbors(last):
            if site in occupied:
                continue
            # Score: number of H residues in chain that would be contacted
            score = 0
            if sequence[i] == 'H':
                for j in range(i - 1, -1, -1):
                    if i - j > 1:
                        break
                    if sequence[j] == 'H':
                        dx = abs(site[0] - chain[j][0])
                        dy = abs(site[1] - chain[j][1])
                        dz = abs(site[2] - chain[j][2])
                        if dx + dy + dz == 1:
                            score += 1
            else:
                # For P residues, still check if placing here enables future contacts
                # by being adjacent to H residues
                for j in range(i - 1, -1, -1):
                    if i - j > 1:
                        break
                    if sequence[j] == 'H':
                        dx = abs(site[0] - chain[j][0])
                        dy = abs(site[1] - chain[j][1])
                        dz = abs(site[2] - chain[j][2])
                        if dx + dy + dz == 1:
                            score += 1
            
            if score > best_score:
                best_score = score
                best_sites = [site]
            elif score == best_score:
                best_sites.append(site)
        
        if not best_sites:
            return None
        
        site = rng.choice(best_sites)
        chain.append(site)
        occupied[site] = i
    
    return chain, occupied

def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive neighbors)."""
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for nb in _neighbors(chain[i]):
            j = pos_to_idx.get(nb)
            if j is not None and j > i + 1:
                contacts += 1
    return -contacts

def _local_search(sequence, chain, occupied, time_budget):
    """
    Fast local search: try moving individual residues to adjacent free sites.
    Only accept moves that improve energy.
    Returns (chain, occupied, energy).
    """
    n = len(chain)
    energy = _compute_energy(sequence, chain)
    
    deadline = time.time() + time_budget
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to escape local optima
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            for nb in _neighbors(chain[i]):
                if nb in occupied:
                    continue
                # Check if moving i to nb is valid (no collision)
                old_pos = chain[i]
                # Remove i from occupied temporarily
                del occupied[old_pos]
                
                # Check if nb is free (should be, but verify)
                if nb in occupied:
                    occupied[old_pos] = i
                    continue
                
                # Tentatively move
                chain[i] = nb
                occupied[nb] = i
                
                # Check self-avoidance: ensure consecutive residues still have unit steps
                valid = True
                if i > 0:
                    d = abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) + abs(chain[i][2] - chain[i-1][2])
                    if d != 1:
                        valid = False
                if i < n - 1:
                    d = abs(chain[i+1][0] - chain[i][0]) + abs(chain[i+1][1] - chain[i][1]) + abs(chain[i+1][2] - chain[i][2])
                    if d != 1:
                        valid = False
                
                if not valid:
                    # Revert
                    chain[i] = old_pos
                    occupied[old_pos] = i
                    continue
                
                new_energy = _compute_energy(sequence, chain)
                if new_energy < energy:
                    energy = new_energy
                    improved = True
                else:
                    # Revert
                    chain[i] = old_pos
                    occupied[old_pos] = i
                    del occupied[nb]
    
    return chain, occupied, energy

def _fold_single(sequence, n_restarts, time_budget, rng):
    """
    Run multiple greedy restarts with local search.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start = time.time()
    deadline = start + time_budget
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        # Greedy growth
        result = _grow(sequence, rng)
        if result is None:
            continue
        chain, occupied = result
        
        # Local search with remaining time
        remaining = deadline - time.time() - 1.0
        if remaining < 0.1:
            break
        
        chain, occupied, energy = _local_search(sequence, chain, occupied, remaining)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    return best_chain if best_chain else [(i, 0, 0) for i in range(n)]

def _fallback_chain(n):
    """Simple straight-line fallback."""
    return [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    """
    Fold an HP sequence on the simple cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    remaining = _time_left()
    if remaining < 1.0:
        return [list(p) for p in _fallback_chain(n)]
    
    # Allocate time for this call: assume up to 26 sequences
    # Use a fraction of remaining time, capped per call
    per_call_budget = min(remaining * 0.9, 30.0)
    
    # Number of restarts: more for shorter chains
    if n <= 48:
        n_restarts = max(5, int(per_call_budget / 2.0))
    elif n <= 80:
        n_restarts = max(3, int(per_call_budget / 4.0))
    else:
        n_restarts = max(2, int(per_call_budget / 6.0))
    
    rng = random.Random(42 + n * 7)
    
    best_chain = _fold_single(sequence, n_restarts, per_call_budget, rng)
    
    # Validate self-avoidance
    if len(set(best_chain)) != n:
        best_chain = _fallback_chain(n)
    
    # Validate unit steps
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = _fallback_chain(n)
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
