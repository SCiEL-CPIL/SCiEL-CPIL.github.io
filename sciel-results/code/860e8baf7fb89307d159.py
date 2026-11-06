import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 330.0
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
    # Map position to index for O(1) lookup
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    # Only check H residues
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
        if _time_left() < 5.0:
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

    # Check 6 neighbors
    for site in _neighbours(curr_pos):
        if site in occupied:
            continue

        # Check connectivity to prev and next
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
        
        # 90 degree rotation logic
        if axis == 0: # x-axis
            if sign == 1:
                new_vec = [vec[0], -vec[2], vec[1]]
            else:
                new_vec = [vec[0], vec[2], -vec[1]]
        elif axis == 1: # y-axis
            if sign == 1:
                new_vec = [vec[2], vec[1], -vec[0]]
            else:
                new_vec = [-vec[2], vec[1], vec[0]]
        else: # z-axis
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

    # Check boundary connectivity after reversal
    # The segment chain[i..j] becomes chain[j..i]
    # New neighbors of chain[i] (now at pos i) are chain[i-1] and chain[j]
    # New neighbors of chain[j] (now at pos j) are chain[i] and chain[j+1]
    
    # Actually, easier to just construct and check set uniqueness + connectivity
    # But construction is O(N).
    
    # Connectivity checks:
    # 1. chain[i-1] must be neighbor of chain[j] (which moves to pos i)
    # 2. chain[j+1] must be neighbor of chain[i] (which moves to pos j)
    
    if i > 0:
        d1 = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(dimension))
        if d1 != 1:
            return None
    if j < n - 1:
        d2 = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(dimension))
        if d2 != 1:
            return None

    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]

    # Check self-avoidance
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
    # Try to improve using Kink and Pivot moves greedily
    greedy_iter = 0
    max_greedy_iters = 2000 if n < 50 else 500
    while greedy_iter < max_greedy_iters and _time_left() > 5.0:
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
        
        if not improved:
            # If no improvement in this iteration, maybe try a few more random kinks
            # or break if stuck
            pass
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    # Phase 2: Simulated Annealing
    T = 2.0
    alpha = 0.99
    min_T = 0.01
    
    while _time_left() > 1.0 and T > min_T:
        # Choose move type
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            # Kink
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, i, dimension)
        elif move_type < 0.9 and n > 15:
            # Pivot
            i = rng.randint(2, n - 2)
            axis = rng.randint(0, dimension - 1)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(current_chain, i, axis, sign, dimension)
        else:
            # Reversal
            i = rng.randint(1, n - 3)
            max_len = min(5, n - i - 1)
            j = rng.randint(i + 1, i + max_len)
            new_chain = _reversal_move(current_chain, i, j, dimension)
        
        if new_chain is None:
            T *= alpha
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= alpha
        
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
    
    # Run beam search
    results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
    
    if not results:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Sort by energy
    results.sort(key=lambda x: x[1])
    
    # Take the best candidate from beam search
    best_chain = results[0][0]
    
    # Run Local Optimization
    best_chain, best_energy = _local_optimize_sa(sequence, best_chain, _time_left())
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
