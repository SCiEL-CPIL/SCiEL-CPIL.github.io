import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

# Pre-define neighbor offsets for 3D lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    
    for i, p in enumerate(chain):
        if is_h[i]:
            x, y, z = p
            if (x+1, y, z) in where:
                j = where[(x+1, y, z)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
            if (x-1, y, z) in where:
                j = where[(x-1, y, z)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
            if (x, y+1, z) in where:
                j = where[(x, y+1, z)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
            if (x, y-1, z) in where:
                j = where[(x, y-1, z)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
            if (x, y, z+1) in where:
                j = where[(x, y, z+1)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
            if (x, y, z-1) in where:
                j = where[(x, y, z-1)]
                if j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _energy_delta(sequence, chain, i, new_pos):
    """Compute the change in energy when moving residue i to new_pos."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    if not is_h[i]:
        return 0
    
    old_pos = chain[i]
    # Remove old position
    where = {}
    for j, p in enumerate(chain):
        if j != i:
            where[p] = j
    
    delta = 0
    # Remove old contacts involving i
    ox, oy, oz = old_pos
    for ox2, oy2, oz2 in _NEIGHBOR_OFFSETS:
        q = (ox+ox2, oy+oy2, oz+oz2)
        if q in where:
            j = where[q]
            if j > i + 1 or (j < i - 1 and abs(j - i) > 1):
                if abs(j - i) > 1 and is_h[j]:
                    delta += 1  # removing a contact increases energy by 1
    # Add new contacts involving i
    nx, ny, nz = new_pos
    for ox2, oy2, oz2 in _NEIGHBOR_OFFSETS:
        q = (nx+ox2, ny+oy2, nz+oz2)
        if q in where:
            j = where[q]
            if abs(j - i) > 1 and is_h[j]:
                delta -= 1  # adding a contact decreases energy by 1
    return delta

def _beam_search(sequence, dimension, beam_width, rng, time_budget):
    """Beam search for folding. Returns list of (chain, energy) for top candidates."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    start = (0, 0, 0)
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break
        
        new_beam = []
        for neg_score, chain, occ, contacts in beam:
            last = chain[-1]
            x, y, z = last
            
            for nx, ny, nz in _NEIGHBOR_OFFSETS:
                site = (x+nx, y+ny, z+nz)
                if site in occ:
                    continue
                
                new_occ = dict(occ)
                new_occ[site] = i
                
                new_contacts = contacts
                if is_h[i]:
                    sx, sy, sz = site
                    for ox, oy, oz in _NEIGHBOR_OFFSETS:
                        q = (sx+ox, sy+oy, sz+oz)
                        if q in new_occ:
                            j = new_occ[q]
                            if j < i - 1 and is_h[j]:
                                new_contacts += 1
                
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ, new_contacts))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results

def _kink_move(chain, i):
    """Try to move residue i to an adjacent free site. Returns new_chain if valid, else None."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None

    occupied = set(chain)
    occupied.discard(chain[i])

    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    curr_pos = chain[i]
    
    cx, cy, cz = curr_pos
    
    for nx, ny, nz in _NEIGHBOR_OFFSETS:
        site = (cx+nx, cy+ny, cz+nz)
        if site in occupied:
            continue
        
        px, py, pz = prev_pos
        if abs(site[0] - px) + abs(site[1] - py) + abs(site[2] - pz) != 1:
            continue
            
        qx, qy, qz = next_pos
        if abs(site[0] - qx) + abs(site[1] - qy) + abs(site[2] - qz) != 1:
            continue
        
        new_chain = list(chain)
        new_chain[i] = site
        return new_chain
    return None

def _pivot_move(chain, i, axis, sign):
    """Perform a pivot move at index i. Returns new_chain if valid, else None."""
    n = len(chain)
    if i <= 1 or i >= n - 1:
        return None
    
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    
    head_set = set(head)
    new_tail = []
    seen_in_tail = set()
    
    px, py, pz = pivot
    
    for pos in tail:
        vx, vy, vz = pos[0] - px, pos[1] - py, pos[2] - pz
        
        if axis == 0:
            if sign == 1:
                nvx, nvy, nvz = vx, -vz, vy
            else:
                nvx, nvy, nvz = vx, vz, -vy
        elif axis == 1:
            if sign == 1:
                nvx, nvy, nvz = vz, vy, -vx
            else:
                nvx, nvy, nvz = -vz, vy, vx
        else:
            if sign == 1:
                nvx, nvy, nvz = -vy, vx, vz
            else:
                nvx, nvy, nvz = vy, -vx, vz
        
        new_pos = (px + nvx, py + nvy, pz + nvz)
        
        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)
        
    return head + new_tail

def _reversal_move(chain, i, j):
    """Reverse the segment from i to j (inclusive). Returns new_chain if valid, else None."""
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
    if i == 0 or j == n - 1:
        return None
    
    # Check that the reversal doesn't create collisions
    # The reversed segment must not collide with itself or the rest of the chain
    fixed = chain[:i] + chain[j+1:]
    fixed_set = set(fixed)
    
    segment = chain[i:j+1]
    reversed_segment = list(reversed(segment))
    
    # Check for self-collision in reversed segment
    seen = set()
    for pos in reversed_segment:
        if pos in seen:
            return None
        if pos in fixed_set:
            return None
        seen.add(pos)
    
    # Check that the connections are valid
    # chain[i-1] to reversed_segment[0] must be adjacent
    if i > 0:
        prev = chain[i-1]
        first = reversed_segment[0]
        if abs(prev[0]-first[0]) + abs(prev[1]-first[1]) + abs(prev[2]-first[2]) != 1:
            return None
    
    # reversed_segment[-1] to chain[j+1] must be adjacent
    if j < n - 1:
        last = reversed_segment[-1]
        next_pos = chain[j+1]
        if abs(last[0]-next_pos[0]) + abs(last[1]-next_pos[1]) + abs(last[2]-next_pos[2]) != 1:
            return None
    
    return fixed[:i] + reversed_segment + fixed[i:]

def _local_optimize_sa(sequence, chain, rng):
    """
    Simulated Annealing with Greedy initialization and mixed moves.
    """
    n = len(chain)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Phase 1: Greedy Descent
    greedy_iter = 0
    max_greedy_iters = 3000 if n < 50 else 800
    while greedy_iter < max_greedy_iters and _time_left() > 5.0:
        improved = False
        greedy_iter += 1
        
        # Try Kink moves
        i = rng.randint(1, n - 2)
        new_chain = _kink_move(current_chain, i)
        if new_chain:
            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        if not improved and n > 10:
            # Try Pivot moves
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        if not improved and n > 15:
            # Try Reversal moves
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 2)
            new_chain = _reversal_move(current_chain, i, j)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        if not improved:
            break
    
    # Phase 2: Simulated Annealing
    T = 2.0
    alpha = 0.997
    min_T = 0.01
    
    while _time_left() > 2.0 and T > min_T:
        move_type = rng.random()
        
        new_chain = None
        if move_type < 0.4:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, i)
        elif move_type < 0.75:
            # Pivot move
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign)
        else:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 2)
            new_chain = _reversal_move(current_chain, i, j)
        
        if new_chain is None:
            T *= alpha
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= alpha
        
        if _time_left() < 2.0:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Choose beam width based on chain length
    if n <= 30:
        beam_width = 2000
    elif n <= 50:
        beam_width = 1000
    elif n <= 80:
        beam_width = 500
    else:
        beam_width = 200
    
    # Multi-restart beam search with varied parameters
    best_chain = None
    best_energy = float('inf')
    
    # Primary beam search
    results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
    
    if results:
        results.sort(key=lambda x: x[1])
        best_chain = results[0][0]
        best_energy = results[0][1]
    
    # Try a second beam search with different width if time permits
    if _time_left() > 30.0 and n <= 60:
        alt_width = max(100, beam_width // 2)
        alt_rng = random.Random(99999 + n)
        alt_results = _beam_search(sequence, dimension, alt_width, alt_rng, _time_left())
        if alt_results:
            alt_results.sort(key=lambda x: x[1])
            if alt_results[0][1] < best_energy:
                best_chain = alt_results[0][0]
                best_energy = alt_results[0][1]
    
    if best_chain is None:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Run local optimization
    opt_rng = random.Random(55555 + n)
    best_chain, best_energy = _local_optimize_sa(sequence, best_chain, opt_rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
