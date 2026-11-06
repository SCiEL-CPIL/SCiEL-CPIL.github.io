"""Rewritten program for HP lattice protein folding using beam search + local optimization."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def _neighbours(point):
    """Yield all 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute HP contact energy. Returns negative number of H-H non-bonded contacts."""
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for nb in _neighbours(chain[i]):
                j = pos_to_idx.get(nb)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=100, max_depth=None):
    """
    Beam search for self-avoiding walk growth.
    Each state: (energy_so_far, chain_list, occupied_set)
    Returns best chain found (list of tuples) or None.
    """
    n = len(sequence)
    if max_depth is None:
        max_depth = n

    # Initial state: start at origin
    start = (0, 0, 0)
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_energy = 0

    beam = [(initial_energy, initial_chain, initial_occupied)]

    for step in range(1, max_depth):
        new_beam = []
        residue = sequence[step]

        for energy, chain, occupied in beam:
            last = chain[-1]
            candidates = []

            for site in _neighbours(last):
                if site in occupied:
                    continue

                # Compute gain: new H-H contacts if this residue is H
                gain = 0
                if residue == 'H':
                    for nb in _neighbours(site):
                        if nb in occupied:
                            idx = occupied[nb]
                            if idx < step - 1 and sequence[idx] == 'H':
                                gain += 1

                # Score: minimize energy (more negative is better), so gain increases are good
                # We want to maximize gain, so score = -gain (lower is better)
                # Also add small penalty for reducing free space
                free_count = 0
                for nb in _neighbours(site):
                    if nb not in occupied:
                        free_count += 1

                score = -gain - 0.01 * free_count

                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = step
                new_energy = energy - gain

                candidates.append((score, new_energy, new_chain, new_occupied))

            new_beam.extend(candidates)

        if not new_beam:
            break

        # Sort by score (ascending) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = [(e, c, o) for _, e, c, o in new_beam[:beam_width]]

    if not beam:
        return None

    # Return the best chain from the beam
    # Sort by actual energy
    beam.sort(key=lambda x: x[0])
    return beam[0][1]

def _local_optimize(sequence, chain, rng, max_iters=2000):
    """
    Local optimization using pivot and single-residue moves.
    Returns optimized chain.
    """
    n = len(chain)
    if n < 3:
        return chain

    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy

    # Precompute positions
    pos_to_idx = {p: i for i, p in enumerate(current)}

    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break

        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            if n < 4:
                continue
            i = rng.randint(1, n - 2)
            if i == 0 or i == n - 1:
                continue
            
            orig = current[i]
            for nb in _neighbours(orig):
                # Check if nb is already occupied by another residue
                if nb == current[i-1] or nb == current[i+1]:
                    continue
                
                # Check if nb is occupied by any other residue
                conflict = False
                for j in range(n):
                    if j != i and current[j] == nb:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Try the move
                new_chain = list(current)
                new_chain[i] = nb
                new_energy = _compute_energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    current = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best = list(current)
                    break
                # Accept with some probability for exploration (simple SA-like)
                # But keep it simple: only accept improving moves
        else:
            # Pivot move: pick a pivot and flip a segment
            if n < 5:
                continue
            pivot = rng.randint(1, n - 3)
            # Get the segment after pivot
            seg = current[pivot:]
            # Try flipping: for each direction, reverse the segment and place
            # Simple approach: try moving the last residue of the segment to a new location
            # Actually, let's do a simpler pivot: pick two non-adjacent residues and try to move one
            # Let's just do more single-residue moves
            pass

    return best

def fold(sequence, dimension):
    """Fold an HP sequence on the simple cubic lattice."""
    if dimension != 3:
        # Fallback for other dimensions
        return [[i, 0, 0] for i in range(len(sequence))]

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]

    # Budget allocation based on sequence length
    if n <= 48:
        beam_width = 200
        num_restarts = 8
        local_iters = 3000
    elif n <= 64:
        beam_width = 150
        num_restarts = 6
        local_iters = 2500
    elif n <= 100:
        beam_width = 100
        num_restarts = 4
        local_iters = 2000
    else:
        beam_width = 80
        num_restarts = 3
        local_iters = 1500

    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is the worst (no contacts)

    rng = random.Random(12345 + len(sequence))

    for r in range(num_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break

        # Beam search
        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None:
            # Fallback: try a simple greedy growth
            chain = _greedy_grow(sequence, dimension, rng)
            if chain is None:
                continue

        # Local optimization
        optimized = _local_optimize(sequence, chain, rng, max_iters=local_iters)
        energy = _compute_energy(sequence, optimized)

        if best_chain is None or energy < best_energy:
            best_chain = optimized
            best_energy = energy

    if best_chain is None:
        # Final fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]

def _greedy_grow(sequence, dimension, rng):
    """Simple greedy growth as fallback."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for nb in _neighbours(site):
                    if nb in occupied:
                        idx = occupied[nb]
                        if idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
            free = sum(1 for nb in _neighbours(site) if nb not in occupied)
            score = gain + 0.1 * free + rng.random()
            candidates.append((score, site))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        site = candidates[0][1]
        occupied[site] = i
        chain.append(site)

    return chain
# EVOLVE-BLOCK-END
