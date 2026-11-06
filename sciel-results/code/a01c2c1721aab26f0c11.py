"""HP lattice protein folding - beam search with local refinement."""
# EVOLVE-BLOCK-START
import random
import time
from heapq import heappush, heappop


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
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_init(sequence, dimension, rng, time_limit=0.5):
    """Greedy chain growth preferring sites with new H-H contacts."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    t0 = time.time()
    for i in range(1, len(sequence)):
        if time.time() - t0 > time_limit:
            return None
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random() * 0.01, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _beam_search(sequence, dimension, width=50, max_time=10.0):
    """Beam search over partial chains, keeping top-width by energy + heuristic."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    t0 = time.time()
    # Each state: (negative_score, chain, occupied_set)
    # Score = energy_so_far + heuristic (more negative is better)
    start = (0,) * dimension
    occupied = {start: 0}
    beam = [(-0.0, [start], occupied)]
    
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        new_beam = []
        for score, chain, occ in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                new_chain = chain + [site]
                new_occ = dict(occ)
                new_occ[site] = i
                
                # Compute incremental energy
                inc_energy = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            inc_energy += 1
                
                # Heuristic: count potential future contacts
                heuristic = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q not in new_occ:
                            heuristic += 0.05  # small bonus for free space near H
                
                new_score = score - inc_energy + heuristic
                new_beam.append((new_score, new_chain, new_occ))
        
        if not new_beam:
            return None
        
        # Keep top-width states
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:width]
    
    # Return best complete chain
    if not beam:
        return None
    best_chain = min(beam, key=lambda x: _energy(sequence, x[1]))[1]
    return best_chain


def _local_search(sequence, chain, dimension, rng, max_time=5.0):
    """Local search with pivot and slide moves."""
    t0 = time.time()
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_energy = best_energy
    n = len(sequence)
    
    while time.time() - t0 < max_time:
        # Try a random move
        move_type = rng.random()
        if move_type < 0.5:
            # Pivot move
            if n <= 2:
                break
            pivot = rng.randint(1, n - 2)
            tail = list(best_chain[pivot + 1:])
            # Try all 6 directions for the first tail segment
            for axis in range(dimension):
                for step in (-1, 1):
                    new_tail = [list(best_chain[pivot + 1])]
                    new_tail[0][axis] += step
                    # Check if this is valid
                    if tuple(new_tail[0]) in {tuple(p) for p in best_chain[:pivot + 1]}:
                        continue
                    # Try to complete the rest
                    valid = True
                    for j in range(1, len(tail)):
                        new_tail.append(list(new_tail[-1]))
                        # Try all 6 directions
                        placed = False
                        for a2 in range(dimension):
                            for s2 in (-1, 1):
                                candidate = list(new_tail[-1])
                                candidate[a2] += s2
                                if tuple(candidate) not in {tuple(p) for p in best_chain[:pivot + 1] + new_tail}:
                                    new_tail[-1] = candidate
                                    placed = True
                                    break
                            if placed:
                                break
                        if not placed:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check self-avoidance for entire chain
                    full_chain = best_chain[:pivot + 1] + new_tail
                    if len(set(map(tuple, full_chain))) != n:
                        continue
                    new_energy = _energy(sequence, full_chain)
                    if new_energy < current_energy or rng.random() < 0.1:
                        best_chain = full_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                        break
        else:
            # Slide move: move a middle segment
            if n <= 3:
                break
            seg_start = rng.randint(1, n - 3)
            seg_len = min(rng.randint(1, 3), n - seg_start - 1)
            segment = best_chain[seg_start:seg_start + seg_len]
            # Try shifting the segment
            for axis in range(dimension):
                for step in (-1, 1):
                    new_segment = []
                    for p in segment:
                        np = list(p)
                        np[axis] += step
                        new_segment.append(tuple(np))
                    # Check validity
                    full_chain = best_chain[:seg_start] + new_segment + best_chain[seg_start + seg_len:]
                    if len(set(map(tuple, full_chain))) != n:
                        continue
                    # Check unit steps
                    valid = True
                    for k in range(n - 1):
                        diff = sum(abs(a - b) for a, b in zip(full_chain[k], full_chain[k + 1]))
                        if diff != 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_energy = _energy(sequence, full_chain)
                    if new_energy < current_energy or rng.random() < 0.1:
                        best_chain = full_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                        break
    
    return best_chain


def _straight_line(sequence, dimension):
    """Fallback: straight line is always self-avoiding."""
    return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]


def fold(sequence, dimension):
    """Best conformation using beam search + local search with time budget."""
    n = len(sequence)
    t0 = time.time()
    deadline = 240.0  # Total budget for all sequences
    margin = 10.0  # Safety margin
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy inits
    time_per_init = min(0.3, (deadline - (time.time() - t0) - margin) / 20)
    for attempt in range(20):
        if time.time() - t0 > deadline - margin:
            break
        chain = _greedy_init(sequence, dimension, rng, time_limit=time_per_init)
        if chain is not None:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain, best_energy = chain, e
    
    # Phase 2: Beam search if time remains
    remaining = deadline - (time.time() - t0) - margin
    if remaining > 2.0 and best_chain is not None:
        beam_time = min(remaining * 0.5, 8.0)
        beam_chain = _beam_search(sequence, dimension, width=30, max_time=beam_time)
        if beam_chain is not None:
            e = _energy(sequence, beam_chain)
            if e < best_energy:
                best_chain, best_energy = beam_chain, e
    
    # Phase 3: Local search refinement
    remaining = deadline - (time.time() - t0) - margin
    if remaining > 1.0 and best_chain is not None:
        ls_time = min(remaining * 0.7, 10.0)
        refined = _local_search(sequence, best_chain, dimension, rng, max_time=ls_time)
        e = _energy(sequence, refined)
        if e < best_energy:
            best_chain, best_energy = refined, e
    
    if best_chain is None:
        best_chain = _straight_line(sequence, dimension)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
