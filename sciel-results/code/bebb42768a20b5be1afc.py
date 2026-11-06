"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 300  # 300s safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _pivot_move(chain, i, rng, dimension):
    """Rotate a segment of the chain by a pivot move. Returns new chain or None if invalid."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    # Choose a random rotation axis and angle
    axis = rng.randint(0, dimension - 1)
    angle = rng.choice([1, -1])  # 90 degree rotation
    
    # Extract the segment to rotate
    pivot = chain[i]
    segment = chain[i + 1:]
    
    # Check if rotation would cause self-avoidance violation
    occupied = {p: idx for idx, p in enumerate(chain[:i + 1])}
    
    # Rotate each point in segment around pivot
    new_segment = []
    for p in segment:
        new_p = list(p)
        # Rotate in the plane perpendicular to axis
        # For 3D, rotate around axis by 90 degrees
        if dimension == 3:
            if axis == 0:
                # Rotate around x-axis: y -> z, z -> -y
                dy = p[1] - pivot[1]
                dz = p[2] - pivot[2]
                new_p[1] = pivot[1] + dz * angle
                new_p[2] = pivot[2] - dy * angle
            elif axis == 1:
                # Rotate around y-axis: x -> z, z -> -x
                dx = p[0] - pivot[0]
                dz = p[2] - pivot[2]
                new_p[0] = pivot[0] + dz * angle
                new_p[2] = pivot[2] - dx * angle
            else:
                # Rotate around z-axis: x -> y, y -> -x
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                new_p[0] = pivot[0] - dy * angle
                new_p[1] = pivot[1] + dx * angle
        else:
            # 2D: rotate in plane
            if axis == 0:
                dy = p[1] - pivot[1]
                new_p[1] = pivot[1] - dy * angle
            else:
                dx = p[0] - pivot[0]
                new_p[0] = pivot[0] + dx * angle
        
        new_p = tuple(new_p)
        
        # Check self-avoidance
        if new_p in occupied and occupied[new_p] != i + len(new_segment):
            return None
        occupied[new_p] = i + len(new_segment)
        new_segment.append(new_p)
    
    return chain[:i + 1] + new_segment


def _local_search(sequence, chain, rng, time_limit):
    """Simulated annealing local search with pivot moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current = chain
    current_energy = _energy(sequence, chain)
    best = list(chain)
    best_energy = current_energy
    
    start_temp = 1.0
    end_temp = 0.01
    total_time = time_limit
    start_time = time.time()
    step = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_time:
            break
        
        # Choose a pivot position
        i = rng.randint(1, n - 2)
        new_chain = _pivot_move(current, i, rng, 3)
        
        if new_chain is None:
            step += 1
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Temperature cooling
        progress = min(1.0, elapsed / total_time)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        step += 1
        if step % 5000 == 0:
            break
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search refinement."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # Allocate time: 60% for greedy growth, 40% for local search
    total_budget = _DEADLINE - time.time()
    greedy_time = total_budget * 0.5
    search_time = total_budget * 0.45
    
    # Phase 1: Generate multiple greedy chains
    greedy_deadline = time.time() + greedy_time
    candidates = []
    for _ in range(50):
        if time.time() > greedy_deadline:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    # Phase 2: Local search on best candidates
    # Sort by energy and take top few
    if candidates:
        candidates.sort(key=lambda c: _energy(sequence, c))
        top_candidates = candidates[:min(5, len(candidates))]
    else:
        top_candidates = []
    
    search_deadline = time.time() + search_time
    for chain in top_candidates:
        if time.time() > search_deadline:
            break
        refined, refined_energy = _local_search(sequence, chain, rng, search_deadline - time.time())
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    # If no valid chain found, use straight line
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
