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

def _get_neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

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
            # Check 6 neighbors
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
            
            # Try 6 neighbors
            for nx, ny, nz in _NEIGHBOR_OFFSETS:
                site = (last[0]+nx, last[1]+ny, last[2]+nz)
                if site in occ:
                    continue
                
                new_occ = dict(occ)
                new_occ[site] = i
                
                new_contacts = contacts
                if is_h[i]:
                    # Check neighbors of the new site for H contacts
                    # Only need to check against previous residues (j < i)
                    # and ensure they are not adjacent in the chain (j != i-1)
                    # Since we add to occ, we can just check all in occ
                    # but we must exclude i-1 which is 'last'
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
        
        # Sort by score (most negative is best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results

def _kink_move(chain, i, dimension):
    """
    Try to move residue i to an adjacent free site.
    Returns new_chain if valid, else None.
    """
    if i <= 0 or i >= len(chain) - 1:
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
        
        # Check distance to prev and next
        px, py, pz = prev_pos
        nx_d = abs(site[0] - px) + abs(site[1] - py) + abs(site[2] - pz)
        if nx_d != 1:
            continue
            
        qx, qy, qz = next_pos
        nq_d = abs(site[0] - qx) + abs(site[1] - qy) + abs(site[2] - qz)
        if nq_d != 1:
            continue
        
        new_chain = list(chain)
        new_chain[i] = site
        return new_chain
    return None

def _pivot_move(chain, i, axis, sign, dimension):
    """
    Perform a pivot move at index i.
    Rotates the tail (i+1 to end) around chain[i] by 90 degrees around 'axis' with 'sign'.
    Returns new_chain if valid, else None.
    """
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

def _local_optimize_sa(sequence, chain, time_budget, rng):
    """
    Simulated Annealing with Greedy initialization and mixed moves.
    """
    n = len(chain)
    dimension = 3
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Phase 1: Greedy Descent
    greedy_iter = 0
    max_greedy_iters = 2000 if n < 50 else 500
    while greedy_iter < max_greedy_iters and _time_left() > 5.0:
        improved = False
        greedy_iter += 1
        
        # Try Kink moves
        i = rng.randint(1, n - 2)
        new_chain = _kink_move(current_chain, i, dimension)
        if new_chain:
            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
        
        if not improved and n > 10:
            # Try Pivot moves
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign, dimension)
            if new_chain:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    # Phase 2: Simulated Annealing
    T = 2.0
    alpha = 0.99
    min_T = 0.01
    
    # Determine number of steps based on time
    # Estimate steps per second
    while _time_left() > 2.0 and T > min_T:
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.6:
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, i, dimension)
        elif move_type < 0.95 and n > 15:
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign, dimension)
        else:
            # Reversal move (less frequent, more expensive)
            i = rng.randint(1, n - 3)
            max_len = min(5, n - i - 1)
            j = rng.randint(i + 1, i + max_len)
            
            # Simple reversal check
            if i > 0:
                d1 = sum(abs(current_chain[i-1][k] - current_chain[j][k]) for k in range(3))
                if d1 != 1:
                    new_chain = None
                else:
                    if j < n - 1:
                        d2 = sum(abs(current_chain[j+1][k] - current_chain[i][k]) for k in range(3))
                        if d2 != 1:
                            new_chain = None
                        else:
                            new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                            if len(set(new_chain)) != len(new_chain):
                                new_chain = None
            else:
                if j < n - 1:
                    d2 = sum(abs(current_chain[j+1][k] - current_chain[i][k]) for k in range(3))
                    if d2 != 1:
                        new_chain = None
                    else:
                        new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                        if len(set(new_chain)) != len(new_chain):
                            new_chain = None
                else:
                    new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                    if len(set(new_chain)) != len(new_chain):
                        new_chain = None

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
        
    return best_chain, best_energy

def _multi_restart_fold(sequence, dimension):
    """
    Run beam search + local optimization multiple times with different seeds/beam widths.
    """
    n = len(sequence)
    best_chain = None
    best_energy = float('inf')
    
    # Determine number of restarts based on time and length
    if n <= 30:
        num_restarts = 5
        beam_widths = [1500, 1000, 800, 600, 400]
    elif n <= 50:
        num_restarts = 4
        beam_widths = [800, 600, 400, 300]
    elif n <= 80:
        num_restarts = 3
        beam_widths = [400, 300, 200]
    else:
        num_restarts = 2
        beam_widths = [200, 150]
        
    # Ensure we don't spend too much time on beam search
    # If time is tight, reduce restarts
    if _time_left() < 50.0:
        num_restarts = min(num_restarts, 2)
    if _time_left() < 20.0:
        num_restarts = 1

    for restart_idx in range(num_restarts):
        if _time_left() < 10.0:
            break
        
        rng = random.Random(12345 + restart_idx * 7919)
        bw = beam_widths[restart_idx % len(beam_widths)]
        
        # Time budget for beam search: allow up to 20% of remaining time or 10s
        bs_budget = min(10.0, _time_left() * 0.2)
        
        results = _beam_search(sequence, dimension, bw, rng, bs_budget)
        
        if not results:
            continue
        
        results.sort(key=lambda x: x[1])
        
        # Take top 2 candidates for local optimization
        for cand_idx in range(min(2, len(results))):
            if _time_left() < 5.0:
                break
            
            chain = results[cand_idx][0]
            opt_chain, opt_energy = _local_optimize_sa(sequence, chain, _time_left(), rng)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
                
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    
    return _multi_restart_fold(sequence, dimension)

# EVOLVE-BLOCK-END
