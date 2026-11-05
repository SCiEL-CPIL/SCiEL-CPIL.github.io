import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta(sequence, chain, i, j, new_pos):
    """
    Compute the change in energy when moving residue i from chain[i] to new_pos,
    given that residue j is at chain[j]. This is an approximation for segment moves.
    For a full segment move, we'd need to recompute all contacts involving the segment.
    """
    # This function is a stub for now; we'll recompute energy for segment moves.
    return 0

def _greedy_grow(sequence, rng=None):
    """Fast greedy growth to get an initial conformation."""
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        last = chain[-1]
        candidates = []
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
            # Also consider future potential: prefer sites with more open neighbors
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            # Combine gain and openness
            score = gain * 10 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        # Pick best candidate
        candidates.sort(key=lambda x: -x[0])
        best_site = candidates[0][1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain, _energy(sequence, chain)

def _local_search(sequence, chain, deadline, rng):
    """
    Local search with segment moves.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 3.0
    cooling = 0.999
    min_T = 0.1
    
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Segment move: pick a segment and try to relocate it
        if n < 4:
            continue
        
        # Pick segment length (shorter segments for more exploration)
        max_seg = min(n // 3, 8)
        seg_len = rng.randint(2, max_seg)
        i = rng.randint(0, n - seg_len)
        j = i + seg_len - 1
        
        head = current_chain[:i]
        tail = current_chain[j+1:]
        
        # Build occupied set for head and tail
        occupied_ht = set(head) | set(tail)
        
        # Find valid start positions for the segment
        anchor = head[-1] if head else (0, 0)
        start_candidates = []
        for nb in _neighbours(anchor):
            if nb not in occupied_ht:
                start_candidates.append(nb)
        
        if not start_candidates:
            continue
        
        start_pos = rng.choice(start_candidates)
        new_segment = [start_pos]
        new_occupied = set(occupied_ht) | {start_pos}
        
        success = True
        for k in range(1, seg_len):
            last_pos = new_segment[-1]
            next_candidates = []
            for nb in _neighbours(last_pos):
                if nb not in new_occupied:
                    next_candidates.append(nb)
            
            if not next_candidates:
                success = False
                break
            
            next_pos = rng.choice(next_candidates)
            new_segment.append(next_pos)
            new_occupied.add(next_pos)
        
        if not success:
            continue
        
        # Check connection to tail
        if tail:
            tail_start = tail[0]
            d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
            if d != 1:
                continue
        
        new_chain = head + new_segment + tail
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718, -delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation
        init_chain, init_energy = _greedy_grow(sequence, rng)
        if init_chain is None:
            # Fallback: straight line
            init_chain = [(i, 0) for i in range(n)]
            init_energy = _energy(sequence, init_chain)
        
        # Run local search
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    n = len(sequence)
    
    # Total budget: 120s for all sequences. Use 10s per sequence to be safe.
    deadline = start_time + 10.0
    
    if n == 0:
        return []
    
    # Number of restarts
    n_restarts = 15
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
