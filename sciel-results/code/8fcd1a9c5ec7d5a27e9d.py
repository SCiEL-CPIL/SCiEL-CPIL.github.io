import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 320.0
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
    
    start = (0,) * dimension
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 10.0:
            break
        
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

    for site in _neighbours(curr_pos):
        if site in occupied:
            continue

        dist_prev = sum(abs(site[k] - prev_pos[k]) for k in range(dimension))
        dist_next = sum(abs(site[k] - next_pos[k]) for k in range(dimension))

        if dist_prev == 1 and dist_next == 1:
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
    
    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        
        if axis == 0:
            if sign == 1:
                new_vec = [vec[0], -vec[2], vec[1]]
            else:
                new_vec = [vec[0], vec[2], -vec[1]]
        elif axis == 1:
            if sign == 1:
                new_vec = [vec[2], vec[1], -vec[0]]
            else:
                new_vec = [-vec[2], vec[1], vec[0]]
        else:
            if sign == 1:
                new_vec = [-vec[1], vec[0], vec[2]]
            else:
                new_vec = [vec[1], -vec[0], vec[2]]
        
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        
        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)
        
    return head + new_tail

def _reversal_move(chain, i, j, dimension):
    """
    Reverse a segment of the chain from i to j (inclusive).
    Returns new_chain if valid, else None.
    """
    n = len(chain)
    if i < 1 or j >= n - 1 or j <= i:
        return None
    
    if i > 0:
        d1 = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(dimension))
        if d1 != 1:
            return None
    if j < n - 1:
        d2 = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(dimension))
        if d2 != 1:
            return None

    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]

    if len(set(new_chain)) != len(new_chain):
        return None

    return new_chain

def _local_optimize_sa(sequence, chain, time_budget):
    """
    Simulated Annealing with Greedy initialization and mixed moves.
    """
    n = len(chain)
    dimension = len(chain[0])
    rng = random.Random(12345 + n)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Phase 1: Greedy Descent
    greedy_iter = 0
    max_greedy_iters = 3000 if n < 50 else 800
    while greedy_iter < max_greedy_iters and _time_left() > 10.0:
        improved = False
        greedy_iter += 1
        
        # Try Kink moves (fast)
        i = rng.randint(1, n - 2)
        new_chain = _kink_move(current_chain, i, dimension)
        if new_chain:
            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
        
        if not improved and n > 10:
            # Try Pivot moves (slower)
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, dimension - 1)
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
    T = 3.0
    alpha = 0.995
    min_T = 0.005
    
    while _time_left() > 2.0 and T > min_T:
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, i, dimension)
        elif move_type < 0.9 and n > 15:
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, dimension - 1)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign, dimension)
        else:
            i = rng.randint(1, n - 3)
            max_len = min(7, n - i - 1)
            j = rng.randint(i + 1, i + max_len)
            new_chain = _reversal_move(current_chain, i, j, dimension)
        
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
    
    # Configure restarts based on time and chain length
    if n <= 30:
        beam_widths = [2000, 1500, 1000]
        seeds = [12345, 54321, 99999]
    elif n <= 50:
        beam_widths = [1000, 800, 600]
        seeds = [12345, 54321, 99999]
    elif n <= 80:
        beam_widths = [500, 400, 300]
        seeds = [12345, 54321]
    else:
        beam_widths = [200, 150]
        seeds = [12345, 54321]
    
    for restart_idx in range(len(seeds)):
        if _time_left() < 15.0:
            break
        
        rng = random.Random(seeds[restart_idx])
        bw = beam_widths[restart_idx % len(beam_widths)]
        
        results = _beam_search(sequence, dimension, bw, rng, _time_left())
        
        if not results:
            continue
        
        results.sort(key=lambda x: x[1])
        
        # Take top 2 candidates for local optimization
        for cand_idx in range(min(2, len(results))):
            if _time_left() < 10.0:
                break
            
            chain = results[cand_idx][0]
            opt_chain, opt_energy = _local_optimize_sa(sequence, chain, _time_left())
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    if best_chain is None:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    
    return _multi_restart_fold(sequence, dimension)

# EVOLVE-BLOCK-END
