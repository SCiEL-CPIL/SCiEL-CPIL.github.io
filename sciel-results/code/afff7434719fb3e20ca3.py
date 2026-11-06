"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, max_time):
    """Beam search over partial chains with incremental energy tracking."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start = tuple([0] * dimension)
    
    # Each beam entry: (chain_tuple, energy, occupied_set, h_positions_frozenset)
    # We use frozenset for h_positions to allow hashing if needed, but we don't need to hash beams
    # Actually, we just need to store them, so regular set is fine
    beams = [( (start,), 0, {start}, frozenset([start]) if sequence[0] == 'H' else frozenset() )]
    
    deadline = time.time() + max_time
    best_chain = None
    best_energy = 0
    
    for step in range(1, n):
        if time.time() > deadline:
            break
        residue = sequence[step]
        new_beams = []
        
        for chain, energy, occupied, h_pos in beams:
            last = chain[-1]
            
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Compute energy gain: number of new H-H contacts
                gain = 0
                if residue == "H":
                    for nb in _neighbours(site):
                        if nb in h_pos and nb != last:
                            gain += 1
                
                new_energy = energy - gain
                new_chain = chain + (site,)
                new_occupied = occupied | {site}
                new_h_pos = h_pos | frozenset([site]) if residue == 'H' else h_pos
                
                new_beams.append((new_chain, new_energy, new_occupied, new_h_pos))
        
        if not new_beams:
            break
        
        # Keep only the best beam_width entries (most negative energy first)
        new_beams.sort(key=lambda x: x[1])
        beams = new_beams[:beam_width]
    
    # Find the best complete chain
    if step == n - 1 and beams:
        for chain, energy, _, _ in beams:
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is not None:
        return [list(p) for p in best_chain]
    
    # Fallback: straight line
    return [[i] + [0] * (dimension - 1) for i in range(n)]


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
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


def fold(sequence, dimension):
    """Find the best conformation using beam search and randomized restarts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    total_budget = 300
    safety_margin = 30
    effective_budget = total_budget - safety_margin
    
    best_chain = None
    best_energy = 0
    
    # Try beam search with increasing beam widths
    beam_configs = [
        (50, 0.3),   # beam width, fraction of time
        (80, 0.3),
        (120, 0.4),
    ]
    
    for bw, time_frac in beam_configs:
        elapsed = time.time() - start_time
        remaining = effective_budget - elapsed
        if remaining < 10:
            break
        
        time_for_this = min(remaining * time_frac, remaining)
        time_for_this = max(time_for_this, 10)
        
        chain = _beam_search(sequence, dimension, bw, time_for_this)
        if chain:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Use remaining time for randomized greedy growths
    elapsed = time.time() - start_time
    remaining = effective_budget - elapsed
    if remaining > 10:
        rng = random.Random(12345 + len(sequence))
        # Estimate how many restarts we can do
        # Each restart takes roughly O(n * 6 * 6) operations, so for n=136, maybe ~1000 ops
        # Let's just run as many as we can
        for _ in range(10000):  # Will break based on time
            if time.time() - start_time > effective_budget:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _compute_energy(sequence, chain)
            if value < best_energy:
                best_energy = value
                best_chain = chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
