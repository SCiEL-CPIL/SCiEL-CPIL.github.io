"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq


_DEADLINE = time.time() + 300  # 300s safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, max_steps):
    """Beam search over self-avoiding walks, scoring by current H-H contacts plus free space."""
    start = (0,) * dimension
    # Each beam entry: (negative_score, chain_tuple, occupied_set)
    # We maximize score = contacts + 0.1 * free_space, so we use negative for min-heap
    initial_chain = (start,)
    initial_occupied = {start: 0}
    initial_score = 0.0
    beam = [(-initial_score, 0, initial_chain, initial_occupied)]
    
    best_chain = None
    best_energy = 0  # 0 contacts initially
    
    for step in range(1, len(sequence)):
        if time.time() > _DEADLINE:
            break
        new_beam = []
        seen = set()
        
        for neg_score, seq_idx, chain, occupied in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                # Check if this site would create immediate self-intersection with non-adjacent
                # (already handled by occupied set)
                
                # Compute score contribution
                gain = 0
                if sequence[step] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < step - 1 and sequence[j] == "H":
                                gain += 1
                
                # Count free neighbors for diversity
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                score = gain + 0.1 * free
                new_chain = chain + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = step
                
                # Use chain hash for deduplication within this step
                key = new_chain
                if key in seen:
                    continue
                seen.add(key)
                
                new_beam.append((-score, step, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        # Sort by score (most negative = best) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        # Track best complete chain when we finish
        if step == len(sequence) - 1:
            for neg_score, seq_idx, chain, occupied in beam:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
    
    if best_chain is not None:
        return list(best_chain), best_energy
    
    # Fallback: return the best from the last beam if we didn't complete
    if beam:
        # Pick the one with most contacts
        best = None
        best_e = 0
        for neg_score, seq_idx, chain, occupied in beam:
            if len(chain) == len(sequence):
                e = _energy(sequence, chain)
                if e < best_e:
                    best_e = e
                    best = chain
        if best:
            return list(best), best_e
    
    return None, 0


def fold(sequence, dimension):
    """Best conformation using beam search."""
    n = len(sequence)
    
    # Adaptive beam width based on sequence length
    if n <= 20:
        beam_width = 500
    elif n <= 40:
        beam_width = 300
    elif n <= 60:
        beam_width = 200
    elif n <= 80:
        beam_width = 150
    else:
        beam_width = 100
    
    # Try beam search
    chain, energy = _beam_search(sequence, dimension, beam_width, n)
    
    if chain is not None:
        return [list(p) for p in chain]
    
    # Fallback: simple greedy growth
    rng = random.Random(12345 + len(sequence))
    start = (0,) * dimension
    best, best_energy = None, 1
    for _ in range(100):
        if time.time() > _DEADLINE:
            break
        chain, occupied = [start], {start: 0}
        valid = True
        for i in range(1, n):
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
                valid = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        if valid:
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
