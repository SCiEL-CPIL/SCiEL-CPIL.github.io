"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

# Global deadline management
_GLOBAL_DEADLINE = None
_GLOBAL_BEST = {}  # sequence -> (energy, chain)


def _get_deadline():
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 280  # 280s safety margin out of 300s
    return _GLOBAL_DEADLINE


def _time_remaining():
    return _get_deadline() - time.time()


def _neighbours(point):
    """Yield 6 neighbours on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -1 per non-adjacent H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Randomized greedy chain growth.
    bias controls how much we prefer H-H contacts vs free space.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                # Count new H-H contacts this site would form
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            # Score: prefer contacts, then free space, then random
            score = bias * gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_iters=200, rng=None):
    """
    Local search using rotations of internal segments.
    Returns improved chain and energy.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 5:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Precompute H positions
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    for it in range(max_iters):
        if _time_remaining() < 2:
            break
        
        # Pick a random internal segment to rotate
        # Segment from i to j where 1 <= i < j <= n-2
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        seg_len = j - i + 1
        
        if seg_len < 3 or seg_len > min(8, n // 3):
            continue
        
        # Try rotating the segment in one of several ways
        # Simple approach: try moving one residue in the segment
        # to a different valid position
        
        # Pick a residue in the segment to try moving
        k = rng.randint(i, j)
        old_pos = current_chain[k]
        
        # Try all 6 neighbours of old_pos that are not in the segment
        for new_pos in _neighbours(old_pos):
            # Check if new_pos is free (not occupied by non-adjacent-in-segment residues)
            # The residues k-1 and k+1 are adjacent to k in the chain
            # So new_pos must not be occupied by any residue except k-1 and k+1
            # Actually, new_pos must not be in the occupied set after removing k
            # And new_pos must be adjacent to both k-1 and k+1 if they exist
            
            # Check adjacency constraints
            valid = True
            if k > 0 and new_pos != current_chain[k-1]:
                # Must be adjacent to k-1
                if abs(new_pos[0] - current_chain[k-1][0]) + abs(new_pos[1] - current_chain[k-1][1]) + abs(new_pos[2] - current_chain[k-1][2]) != 1:
                    valid = False
            if valid and k < n - 1 and new_pos != current_chain[k+1]:
                if abs(new_pos[0] - current_chain[k+1][0]) + abs(new_pos[1] - current_chain[k+1][1]) + abs(new_pos[2] - current_chain[k+1][2]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Check if new_pos is unoccupied (excluding k itself)
            occ_idx = occupied.get(new_pos)
            if occ_idx is not None and occ_idx != k:
                continue
            
            # Apply move
            current_chain[k] = new_pos
            # Update occupied
            del occupied[old_pos]
            occupied[new_pos] = k
            
            # Compute new energy
            new_energy = _energy(sequence, current_chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                # Revert
                current_chain[k] = old_pos
                occupied[old_pos] = k
                del occupied[new_pos]
        
        if _time_remaining() < 2:
            break
    
    return current_chain, current_energy


def _simulated_annealing(sequence, chain, max_iters=500, rng=None):
    """
    Simple simulated annealing on the chain.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 5:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    for it in range(max_iters):
        if _time_remaining() < 2:
            break
        
        # Pick a random internal residue to move
        k = rng.randint(1, n - 2)
        old_pos = current_chain[k]
        
        # Try a random neighbour
        candidates = list(_neighbours(old_pos))
        rng.shuffle(candidates)
        moved = False
        
        for new_pos in candidates:
            # Check adjacency to neighbours in chain
            valid = True
            if k > 0:
                if abs(new_pos[0] - current_chain[k-1][0]) + abs(new_pos[1] - current_chain[k-1][1]) + abs(new_pos[2] - current_chain[k-1][2]) != 1:
                    valid = False
            if valid and k < n - 1:
                if abs(new_pos[0] - current_chain[k+1][0]) + abs(new_pos[1] - current_chain[k+1][1]) + abs(new_pos[2] - current_chain[k+1][2]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Check if new_pos is unoccupied
            occ_idx = occupied.get(new_pos)
            if occ_idx is not None and occ_idx != k:
                continue
            
            # Apply move
            current_chain[k] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = k
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                moved = True
                break
            else:
                # Revert
                current_chain[k] = old_pos
                occupied[old_pos] = k
                del occupied[new_pos]
        
        T = max(T_min, T * cooling)
        
        if _time_remaining() < 2:
            break
    
    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Fold the given HP sequence on the cubic lattice.
    Returns a list of lattice points (lists of ints).
    """
    global _GLOBAL_BEST
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Check if we have a cached result
    if sequence in _GLOBAL_BEST:
        return [list(p) for p in _GLOBAL_BEST[sequence][1]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (unfolded)
    
    # Phase 1: Multiple greedy growths with different biases
    biases = [0.5, 1.0, 2.0, 3.0, 5.0]
    num_greedy = 50
    
    start_time = time.time()
    for i in range(num_greedy):
        if _time_remaining() < 5:
            break
        bias = biases[i % len(biases)]
        chain = _greedy_grow(sequence, dimension, rng, bias)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    if _time_remaining() > 10:
        chain, energy = _local_search(sequence, best_chain, max_iters=300, rng=rng)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Phase 3: More greedy growths if time allows, then local search
    remaining = _time_remaining()
    if remaining > 20:
        # Estimate time per greedy grow
        elapsed = time.time() - start_time
        avg_time = elapsed / max(1, num_greedy)
        # How many more greedy grows can we do?
        max_additional = int((remaining - 30) / max(avg_time, 0.01))
        max_additional = min(max_additional, 100)  # Cap at 100 additional
        
        for i in range(max_additional):
            if _time_remaining() < 10:
                break
            chain = _greedy_grow(sequence, dimension, rng, biases[i % len(biases)])
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
                # Do a quick local search on the new best
                if _time_remaining() > 5:
                    chain2, energy2 = _local_search(sequence, best_chain, max_iters=100, rng=rng)
                    if energy2 < best_energy:
                        best_energy = energy2
                        best_chain = chain2
    
    # Phase 4: Final simulated annealing if time allows
    if _time_remaining() > 15:
        chain, energy = _simulated_annealing(sequence, best_chain, max_iters=300, rng=rng)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Cache the result
    _GLOBAL_BEST[sequence] = (best_energy, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
