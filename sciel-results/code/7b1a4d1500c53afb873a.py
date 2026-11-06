"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search with incremental contact tracking + local optimization.
Deadline guard for 330s wall-clock limit.
"""
import time
import random
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 330.0  # wall-clock seconds
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, rng, time_budget):
    """Beam search for folding. Returns list of (chain, energy) for top candidates."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    # State: (negative_score, chain, occupied_set, current_contacts)
    # We want to maximize contacts, so minimize negative contacts
    start = (0,) * dimension
    start_occupied = {start}
    start_chain = [start]
    
    # Initial beam
    beam = [(0, start_chain, start_occupied, 0)]  # (neg_score, chain, occupied, contacts)
    
    for i in range(1, n):
        new_beam = []
        for neg_score, chain, occupied, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_occupied = occupied | {site}
                new_contacts = contacts
                if is_h[i]:
                    # Count new H-H contacts with already placed H residues
                    for q in _neighbours(site):
                        if q in occupied:
                            j = chain.index(q) if q in chain else None
                            # More efficient: track positions
                            pass
                    # Better: check each neighbor
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find index of q in chain
                            # This is O(n), let's use a dict
                            pass
                # Let me restructure to use a position dict
                break
            # This approach is too slow. Let me use a different structure.
        break
    
    # Redo with proper structure
    # State: (neg_score, chain_list, occupied_dict, current_contacts)
    # occupied_dict maps point -> index
    
    start = (0,) * dimension
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break  # Return what we have
        
        new_beam = []
        for neg_score, chain, occ, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                new_occ = dict(occ)
                new_occ[site] = i
                new_contacts = contacts
                if is_h[i]:
                    for q in _neighbours(site):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_contacts += 1
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ, new_contacts))
        
        if not new_beam:
            break
        
        # Keep top beam_width by neg_score (most negative = best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    # Convert to results
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results


def _local_optimize(sequence, chain, max_iters=200):
    """Try single-residue moves to improve energy."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):  # Don't move endpoints
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                # Check if new_pos is unoccupied
                occupied = set(chain)
                occupied.discard(old_pos)
                if new_pos in occupied:
                    continue
                # Check if move is valid (unit step to neighbors)
                if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) + abs(new_pos[2] - chain[i-1][2]) != 1:
                    continue
                if i + 1 < n:
                    if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) + abs(new_pos[2] - chain[i+1][2]) != 1:
                        continue
                # Compute new energy
                new_chain = chain[:i] + [new_pos] + chain[i+1:]
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return chain, current_energy


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Choose beam width based on chain length
    if n <= 30:
        beam_width = 1000
    elif n <= 50:
        beam_width = 500
    elif n <= 80:
        beam_width = 300
    else:
        beam_width = 150
    
    # Run beam search
    results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
    
    if not results:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Take top candidates and locally optimize
    best_chain = None
    best_energy = 0
    
    # Sort by energy (most negative first)
    results.sort(key=lambda x: x[1])
    
    # Optimize top 10 candidates
    for chain, energy in results[:10]:
        if _time_left() < 5.0:
            break
        opt_chain, opt_energy = _local_optimize(sequence, chain)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    if best_chain is None:
        best_chain = results[0][0]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
