"""HP lattice protein folding - stronger algorithm with local search and simulated annealing."""
import time
import random
import math

# Deadline guard: track start time and safety margin
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    return 330.0 - _get_start_time() - _DEADLINE_MARGIN

def _is_deadline_exceeded():
    return _time_remaining() <= 0

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP energy: negative count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
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

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied[site] = i
                placed = True
                break
        if not placed:
            return None
    return chain

def _backtrack_move(sequence, chain, dimension, rng):
    """Attempt a backtracking move: pick a random segment and try to move it."""
    n = len(sequence)
    if n < 5:
        return None
    
    # Pick a random segment [i, j] to move
    i = rng.randint(1, n - 2)
    max_len = min(20, n - i - 1)
    seg_len = rng.randint(1, max_len)
    j = min(i + seg_len, n - 1)
    
    # Try to move the segment to a new position
    segment = chain[i:j+1]
    # Remove segment from occupied
    occupied = {p: idx for idx, p in enumerate(chain)}
    for idx in range(i, j+1):
        del occupied[chain[idx]]
    
    # Try to place segment starting from a neighbor of chain[i-1]
    start_site = chain[i-1]
    neighbors = list(_neighbours(start_site))
    rng.shuffle(neighbors)
    
    for new_start in neighbors:
        if new_start in occupied:
            continue
        # Try to grow the segment from new_start
        new_segment = [new_start]
        new_occupied = set(occupied.keys())
        new_occupied.add(new_start)
        success = True
        for k in range(1, len(segment)):
            # The direction from segment[k-1] to segment[k]
            dx = [segment[k][d] - segment[k-1][d] for d in range(dimension)]
            candidate = tuple(new_segment[-1][d] + dx[d] for d in range(dimension))
            if candidate in new_occupied or candidate == chain[i-1]:
                success = False
                break
            new_segment.append(candidate)
            new_occupied.add(candidate)
        if not success:
            continue
        # Check if the end connects to chain[j+1]
        if j + 1 < n:
            end_neighbor_ok = False
            for nb in _neighbours(new_segment[-1]):
                if nb == chain[j+1]:
                    end_neighbor_ok = True
                    break
            if not end_neighbor_ok:
                continue
        # Build new chain
        new_chain = chain[:i] + new_segment + chain[j+1:]
        # Verify self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        return new_chain
    return None

def _local_search(sequence, chain, dimension, rng, max_iters=1000):
    """Local search using backtracking moves with simulated annealing."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        new_chain = _backtrack_move(sequence, current_chain, dimension, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        T = max(T * cooling, T_min)
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=5000):
    """Simulated annealing with backtracking moves."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.5
    cooling = 0.9995
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        new_chain = _backtrack_move(sequence, current_chain, dimension, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        T = max(T * cooling, T_min)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP chain on the cubic lattice using multiple strategies."""
    _get_start_time()
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # start at 0 (unfolded)
    
    # Strategy 1: Multiple greedy initializations
    n_greedy = max(50, min(500, n * 2))
    for _ in range(n_greedy):
        if _is_deadline_exceeded():
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no valid chain found, use a straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Strategy 2: Local search from best greedy result
    if not _is_deadline_exceeded():
        ls_chain, ls_energy = _local_search(sequence, best_chain, dimension, rng, max_iters=500)
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    # Strategy 3: More greedy restarts + local search
    remaining_budget = _time_remaining()
    if remaining_budget > 10:
        n_more = max(20, min(200, int(remaining_budget * 5)))
        for _ in range(n_more):
            if _is_deadline_exceeded():
                break
            chain = _greedy_grow(sequence, dimension, rng)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            # Quick local search
            if not _is_deadline_exceeded():
                ls_chain, ls_energy = _local_search(sequence, chain, dimension, rng, max_iters=200)
                if ls_energy < best_energy:
                    best_energy = ls_energy
                    best_chain = ls_chain
    
    # Strategy 4: Simulated annealing on the best chain
    if not _is_deadline_exceeded() and _time_remaining() > 20:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, dimension, rng, max_iters=3000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Final: ensure we return a valid chain
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
