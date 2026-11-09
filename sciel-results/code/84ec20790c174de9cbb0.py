"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = [time.time()]
_DEADLINE = 205.0  # Safety margin before 210s kill
_initialized = [False]


def _remaining_time():
    if not _initialized[0]:
        _START_TIME[0] = time.time()
        _initialized[0] = True
    return _DEADLINE - (time.time() - _START_TIME[0])


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
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Count potential future H neighbors (lookahead)
            future_h = 0
            if sequence[i] == "P" or i > n - 5:
                # For P or near end, just use free space
                pass
            else:
                for q in _neighbours(site):
                    if q not in occupied:
                        # Check if future residues might be H
                        future_h += 0.3
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Scale gain more aggressively for H residues
            weight = 2.0 if sequence[i] == "H" else 1.0
            score = weight * gain + 0.05 * free + 0.1 * future_h + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _reverse_move(chain, i, j):
    """Try reversing the segment from i to j (inclusive). Return new chain or None."""
    if j - i < 2:
        return None
    # Only need to check the boundary connections and internal self-avoidance of reversed segment
    # The internal steps of the reversed segment are still unit steps (reversal preserves adjacency)
    # Check: chain[i-1] to chain[j] (if i > 0) and chain[i] to chain[j+1] (if j < len-1)
    # And that the reversed segment doesn't collide with itself or with outside parts
    
    # Build set of positions in the reversed segment
    rev_positions = set(chain[i:j+1][::-1])
    
    # Check collision with outside (positions not in [i,j])
    for k in range(len(chain)):
        if i <= k <= j:
            continue
        if chain[k] in rev_positions:
            return None
    
    # Check boundary connections
    if i > 0:
        a, b = chain[i-1], chain[j]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    if j < len(chain) - 1:
        a, b = chain[i], chain[j+1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    
    # Check self-avoidance within reversed segment (should be fine since original was self-avoiding)
    # But verify no duplicates in the new arrangement
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    return new_chain


def _rotate_move(chain, i, j, rng):
    """Try rotating the segment from i to j by 90 degrees around chain[i]. Return new chain or None."""
    if j - i < 2:
        return None
    center = chain[i]
    # Rotation directions: 90, 180, 270 degrees
    for angle in (1, 2, 3):
        new_chain = chain[:i]
        for k in range(i, j+1):
            dx = chain[k][0] - center[0]
            dy = chain[k][1] - center[1]
            if angle == 1:  # 90 deg CCW
                nx, ny = -dy, dx
            elif angle == 2:  # 180 deg
                nx, ny = -dx, -dy
            else:  # 270 deg CCW
                nx, ny = dy, -dx
            new_chain.append((center[0] + nx, center[1] + ny))
        new_chain.extend(chain[j+1:])
        # Check self-avoidance and unit steps
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        if not valid:
            continue
        for k in range(len(new_chain) - 1):
            diff = abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1])
            if diff != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _local_search(sequence, chain, max_iter=5000):
    """Greedy local search: only accept improving moves."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    rng = random.Random(42)
    
    for _ in range(max_iter):
        improved = False
        # Try short reversals and local rotations
        for _ in range(10):
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + 8, n - 1))
            new_chain = _reverse_move(chain, i, j)
            if new_chain is None:
                continue
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                improved = True
                break
            if _remaining_time() < 2.0:
                return best_chain, best_energy
        if not improved:
            break
    return best_chain, best_energy


def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing to improve a chain."""
    start_time = time.time()
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    temperature = 3.0
    cooling = 0.998
    
    while time.time() - start_time < time_budget and temperature > 0.05:
        # Choose move type with bias toward shorter segments for efficiency
        move_type = rng.random()
        if move_type < 0.7:
            # Reverse move - bias toward shorter segments
            max_len = min(n - 1, max(3, int(n * 0.3)))
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + max_len, n - 1))
            new_chain = _reverse_move(chain, i, j)
        else:
            # Rotate move
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + 10, n - 1))
            new_chain = _rotate_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (temperature > 0.1 and rng.random() < exp(-delta / temperature)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        temperature *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing and local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Time allocation: use more time for longer sequences
    base_time = 3.0
    scale = max(1, n // 15)
    total_budget = min(_remaining_time() - 5.0, base_time * scale)
    
    if total_budget < 1.0:
        # Not enough time for annealing, just do greedy
        best, best_energy = None, 1
        for _ in range(30):
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
        if best is None:
            best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Generate initial chains and anneal the best ones
    num_growths = min(30, max(8, n // 3))
    candidates = []
    for _ in range(num_growths):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Evaluate and sort candidates
    candidates_with_energy = []
    for chain in candidates:
        e = _energy(sequence, chain)
        candidates_with_energy.append((e, chain))
    candidates_with_energy.sort(key=lambda x: x[0])
    
    # Anneal top candidates with time budget
    num_to_anneal = min(len(candidates_with_energy), max(1, int(total_budget / 2.5)))
    time_per_anneal = total_budget / max(1, num_to_anneal)
    
    best_chain = candidates_with_energy[0][1]
    best_energy = candidates_with_energy[0][0]
    
    for idx in range(num_to_anneal):
        if _remaining_time() < 8.0:
            break
        e, chain = candidates_with_energy[idx]
        if e >= best_energy and idx > 0:
            break  # No point annealing worse candidates
        
        anneal_budget = min(time_per_anneal, _remaining_time() - 8.0)
        if anneal_budget < 0.5:
            break
        
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, anneal_budget)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Final local search polish
    if _remaining_time() > 3.0:
        polish_budget = min(_remaining_time() - 3.0, 2.0)
        max_iter = int(polish_budget * 5000)
        best_chain, best_energy = _local_search(sequence, best_chain, max_iter)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
