"""HP lattice protein folding on the square lattice: randomized greedy growth + local refinement."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_full(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                contacts += 1
    return -contacts


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_refine(sequence, chain, rng, max_iters=500):
    """Improve a conformation via random 1-move and 2-move local perturbations."""
    n = len(chain)
    if n < 4:
        return chain
    best_chain = list(chain)
    best_energy = _energy_full(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        if _remaining() < 2.0:
            break
        
        # Try a random 1-move: pick a random residue and try to move it to an adjacent free site
        i = rng.randint(1, n - 2)  # don't move endpoints
        old_pos = best_chain[i]
        candidates = [p for p in _neighbours(old_pos) if p not in occupied or occupied[p] == i]
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        if new_pos == old_pos:
            continue
        
        # Check if moving i to new_pos keeps self-avoidance
        # The site new_pos must be free (not occupied by any other residue)
        if new_pos in occupied and occupied[new_pos] != i:
            continue
        
        # Apply move
        del occupied[old_pos]
        occupied[new_pos] = i
        best_chain[i] = new_pos
        
        new_energy = _energy_full(sequence, best_chain)
        
        if new_energy <= best_energy:
            best_energy = new_energy
        else:
            # Revert
            best_chain[i] = old_pos
            del occupied[new_pos]
            occupied[old_pos] = i
    
    return best_chain


def _two_move_refine(sequence, chain, rng, max_iters=200):
    """Improve via random 2-move perturbations (move two non-adjacent residues)."""
    n = len(chain)
    if n < 6:
        return chain
    best_chain = list(chain)
    best_energy = _energy_full(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        if _remaining() < 2.0:
            break
        
        # Pick two non-adjacent residues
        i = rng.randint(1, n - 2)
        j = rng.randint(1, n - 2)
        if abs(i - j) <= 1:
            continue
        
        old_i = best_chain[i]
        old_j = best_chain[j]
        
        new_i_candidates = [p for p in _neighbours(old_i) if p not in occupied or occupied[p] == i]
        new_j_candidates = [p for p in _neighbours(old_j) if p not in occupied or occupied[p] == j]
        
        if not new_i_candidates or not new_j_candidates:
            continue
        
        new_i = rng.choice(new_i_candidates)
        new_j = rng.choice(new_j_candidates)
        
        if new_i == old_i and new_j == old_j:
            continue
        
        # Check validity
        valid = True
        if new_i in occupied and occupied[new_i] not in (i, j):
            valid = False
        if new_j in occupied and occupied[new_j] not in (i, j):
            valid = False
        # After moving i to new_i, check if new_j is still free
        if valid and new_j == old_i and new_i != old_j:
            pass  # new_j is old_i which will be freed
        if valid and new_i == old_j and new_j != old_i:
            pass  # new_i is old_j which will be freed
        
        if not valid:
            continue
        
        # Apply both moves
        del occupied[old_i]
        del occupied[old_j]
        best_chain[i] = new_i
        best_chain[j] = new_j
        occupied[new_i] = i
        occupied[new_j] = j
        
        new_energy = _energy_full(sequence, best_chain)
        
        if new_energy <= best_energy:
            best_energy = new_energy
        else:
            # Revert
            best_chain[i] = old_i
            best_chain[j] = old_j
            del occupied[new_i]
            del occupied[new_j]
            occupied[old_i] = i
            occupied[old_j] = j
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation via multiple randomized greedy growths + local refinement."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    n = len(sequence)
    # Determine number of restarts based on sequence length and time budget
    # Shorter sequences can afford more restarts
    if n <= 20:
        num_restarts = 500
    elif n <= 40:
        num_restarts = 300
    elif n <= 60:
        num_restarts = 150
    elif n <= 80:
        num_restarts = 80
    else:
        num_restarts = 50
    
    for attempt in range(num_restarts):
        if _remaining() < 3.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local refinement
        chain = _local_refine(sequence, chain, rng, max_iters=200)
        
        value = _energy_full(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
            
            # If we found a very good energy, do more refinement
            if best_energy <= -5:
                chain = _two_move_refine(sequence, best_chain, rng, max_iters=100)
                value = _energy_full(sequence, chain)
                if value < best_energy:
                    best_chain = chain
                    best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
