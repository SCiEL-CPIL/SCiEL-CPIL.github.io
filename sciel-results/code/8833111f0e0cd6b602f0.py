"""HP lattice protein folding on the cubic lattice - simulated annealing approach."""
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


def _initial_fold(sequence, dimension, rng):
    """Create an initial self-avoiding walk using a random greedy approach."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    # Start with a straight line, then perturb
    chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Try to make it more compact by random local moves
    for _ in range(n * 5):
        i = rng.randint(1, n - 2)  # don't move endpoints
        current = chain[i]
        for axis in range(dimension):
            for step in (-1, 1):
                new_pos = list(current)
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                if new_pos not in occupied:
                    # Check it doesn't create adjacency issues (it won't for single residue move)
                    chain[i] = new_pos
                    occupied.pop(current)
                    occupied[new_pos] = i
                    break
        else:
            continue
        break
    
    return chain


def _compute_energy(sequence, chain):
    """Compute the energy (negative number of H-H non-bonded contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _move_residue(chain, i, dimension, rng):
    """Try to move residue i to a random neighboring position."""
    current = chain[i]
    neighbors = list(_neighbours(current))
    rng.shuffle(neighbors)
    
    occupied = {p: idx for idx, p in enumerate(chain)}
    
    for new_pos in neighbors:
        if new_pos not in occupied:
            # Check that moving doesn't create a bond violation
            # The new position must not be adjacent to chain[i-1] or chain[i+1] unless it's the same
            # Actually, for a simple move, we just need new_pos not to be occupied
            # But we also need to ensure the chain remains connected (unit steps)
            # Moving a single residue: check that new_pos is adjacent to both neighbors
            # Wait, no - we're just moving one residue, so we need:
            # 1. new_pos not occupied
            # 2. chain[i-1] adjacent to new_pos (if i > 0)
            # 3. new_pos adjacent to chain[i+1] (if i < len-1)
            
            if i > 0:
                prev = chain[i - 1]
                if sum(abs(a - b) for a, b in zip(prev, new_pos)) != 1:
                    continue
            if i < len(chain) - 1:
                nxt = chain[i + 1]
                if sum(abs(a - b) for a, b in zip(new_pos, nxt)) != 1:
                    continue
            
            return new_pos
    return None


def _flip_segment(chain, i, j, dimension, rng):
    """Try to flip a segment of the chain from i to j."""
    if i >= j:
        return None
    
    occupied = {p: idx for idx, p in enumerate(chain)}
    
    # Remove segment
    segment = chain[i:j + 1]
    for p in segment:
        del occupied[p]
    
    # Try reversed segment
    reversed_segment = list(reversed(segment))
    
    # Check if reversed segment fits
    valid = True
    for p in reversed_segment:
        if p in occupied:
            valid = False
            break
    
    if not valid:
        return None
    
    # Check connectivity at boundaries
    if i > 0:
        prev = chain[i - 1]
        if sum(abs(a - b) for a, b in zip(prev, reversed_segment[0])) != 1:
            return None
    if j < len(chain) - 1:
        nxt = chain[j + 1]
        if sum(abs(a - b) for a, b in zip(reversed_segment[-1], nxt)) != 1:
            return None
    
    # Also check internal connectivity of reversed segment
    for k in range(len(reversed_segment) - 1):
        if sum(abs(a - b) for a, b in zip(reversed_segment[k], reversed_segment[k + 1])) != 1:
            return None
    
    return reversed_segment


def _simulate_annealing(sequence, dimension, rng, time_budget):
    """Simulated annealing to find a good fold."""
    n = len(sequence)
    if n <= 2:
        return [(0,) * dimension, (1, 0, 0) if dimension >= 2 else (1,)]
    
    # Initial fold
    chain = _initial_fold(sequence, dimension, rng)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    t0 = max(0.5, n * 0.1)
    t_end = 0.01
    total_iterations = max(1000, n * 20)
    
    start_time = time.time()
    max_time = time_budget
    
    for it in range(total_iterations):
        if time.time() - start_time > max_time:
            break
        
        # Temperature
        progress = it / total_iterations
        temperature = t0 * math.exp(-10 * progress)
        
        # Choose move type
        move_type = rng.random()
        
        new_chain = None
        delta_energy = 0
        
        if move_type < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            new_pos = _move_residue(chain, i, dimension, rng)
            if new_pos is not None:
                old_pos = chain[i]
                chain[i] = new_pos
                new_energy = _compute_energy(sequence, chain)
                delta_energy = new_energy - current_energy
                if delta_energy > 0 and rng.random() < math.exp(-delta_energy / temperature):
                    current_energy = new_energy
                else:
                    chain[i] = old_pos
        elif move_type < 0.8:
            # Segment flip
            length = rng.randint(2, min(6, n - 1))
            i = rng.randint(0, n - length - 1)
            j = i + length
            new_segment = _flip_segment(chain, i, j, dimension, rng)
            if new_segment is not None:
                old_segment = chain[i:j + 1]
                chain[i:j + 1] = new_segment
                new_energy = _compute_energy(sequence, chain)
                delta_energy = new_energy - current_energy
                if delta_energy > 0 and rng.random() < math.exp(-delta_energy / temperature):
                    current_energy = new_energy
                else:
                    chain[i:j + 1] = old_segment
        else:
            # Reinitialize a portion
            i = rng.randint(0, n // 2)
            j = rng.randint(n // 2, n - 1)
            if i < j:
                # Just do a series of single moves in that region
                for _ in range(3):
                    k = rng.randint(i, min(j, n - 2))
                    if k >= 1:
                        new_pos = _move_residue(chain, k, dimension, rng)
                        if new_pos is not None:
                            old_pos = chain[k]
                            chain[k] = new_pos
                            new_energy = _compute_energy(sequence, chain)
                            delta_energy = new_energy - current_energy
                            if delta_energy > 0 and rng.random() < math.exp(-delta_energy / temperature):
                                current_energy = new_energy
                            else:
                                chain[k] = old_pos
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7)
    
    # Time budget per sequence
    total_budget = 240 / 26  # ~9.2 seconds per sequence
    time_per_run = total_budget / 3  # 3 restarts
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (worst case for negative energy)
    
    for restart in range(3):
        chain = _simulate_annealing(sequence, dimension, rng, time_per_run)
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
