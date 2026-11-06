"""HP lattice protein folding using simulated annealing with reversals and moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_index(sequence, chain):
    """Build a mapping from lattice point to residue index, and a set of H positions."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(tuple(p))
    return where, h_positions


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(tuple(q))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] * n if dimension == 2 else [(0,) * dimension] * n
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(1, n):
        random.shuffle(directions)
        placed = False
        for dx, dy in directions:
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                break
        if not placed:
            return None
    
    return chain


def _reverse_segment(chain, i, j):
    """Reverse the segment from index i to j (inclusive)."""
    if i >= j:
        return chain
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    return new_chain


def _is_valid(chain):
    """Check if the chain is self-avoiding."""
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    return True


def _move_segment(chain, i, j, dx, dy):
    """Move the segment from i to j by (dx, dy)."""
    if i >= j:
        return chain
    new_chain = chain[:i] + [(p[0]+dx, p[1]+dy) for p in chain[i:j+1]] + chain[j+1:]
    return new_chain


def _anneal(sequence, chain, time_budget=10.0):
    """Simulated annealing to improve the conformation."""
    n = len(sequence)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    max_time = time_budget
    
    # Initial temperature
    T = max(1.0, abs(current_energy) / max(1, n // 10))
    cooling_rate = 0.999
    
    while time.time() - start_time < max_time:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Reversal move
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            new_chain = _reverse_segment(current_chain, i, j)
        else:
            # Move a segment
            i = random.randint(0, n - 3)
            j = random.randint(i + 1, n - 2)
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            if dx == 0 and dy == 0:
                continue
            new_chain = _move_segment(current_chain, i, j, dx, dy)
        
        if not _is_valid(new_chain):
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept or reject
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    return best_chain


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 2:
        return [[0] * dimension for _ in range(n)]
    
    global_start = time.time()
    time_limit = 180.0
    
    best_chain = None
    best_energy = 1  # Energy is negative, so 1 is a high (bad) value
    
    # First, try greedy growth to get a good initial solution
    rng = random.Random(12345 + len(sequence))
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Now, use simulated annealing to improve
    # Allocate remaining time for annealing
    elapsed = time.time() - global_start
    remaining_time = time_limit - elapsed - 10.0  # 10s safety margin
    
    if remaining_time <= 0:
        return [list(p) for p in best_chain]
    
    # Do multiple annealing runs
    num_anneals = max(1, int(remaining_time / 5.0))
    
    for anneal_idx in range(num_anneals):
        if time.time() - global_start > time_limit - 15.0:
            break
        
        # Start from best_chain or a new random walk
        if anneal_idx == 0:
            start_chain = best_chain
        else:
            # Try a few random walks and pick the best
            candidates = []
            for _ in range(5):
                rw = _random_walk(sequence, dimension)
                if rw is not None:
                    e = _energy(sequence, rw)
                    candidates.append((e, rw))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                start_chain = candidates[0][1]
            else:
                start_chain = best_chain
        
        # Anneal
        time_for_this_anneal = min(10.0, max(1.0, (time_limit - 15.0 - (time.time() - global_start)) / max(1, num_anneals - anneal_idx)))
        improved = _anneal(sequence, start_chain, time_budget=time_for_this_anneal)
        
        improved_energy = _energy(sequence, improved)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
