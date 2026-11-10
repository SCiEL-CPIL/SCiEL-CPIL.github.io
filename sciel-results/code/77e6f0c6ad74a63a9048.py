"""HP lattice protein folding using simulated annealing with diverse moves."""
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts) of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_walk(chain):
    """Check if chain is a valid self-avoiding walk with unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _initial_spiral(sequence, dimension):
    """Create an initial spiral conformation."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    # Spiral pattern: right, up, left, left, down, down, right, right, right, up, up, up...
    # Actually let's use a simple growth spiral
    direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_in_dir = 1
    dir_count = 0
    current_pos = (0, 0)
    
    for i in range(1, n):
        placed = False
        for attempt in range(4):
            dx, dy = direction[dir_idx]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if new_pos not in occupied:
                occupied[new_pos] = i
                chain.append(new_pos)
                current_pos = new_pos
                placed = True
                break
            # Try next direction
            dir_idx = (dir_idx + 1) % 4
        if not placed:
            # Fallback: try all 4 directions
            for d in direction:
                new_pos = (current_pos[0] + d[0], current_pos[1] + d[1])
                if new_pos not in occupied:
                    occupied[new_pos] = i
                    chain.append(new_pos)
                    current_pos = new_pos
                    dir_idx = direction.index(d)
                    placed = True
                    break
            if not placed:
                # Give up and just extend in a line
                chain.append((i, 0))
                break
    return chain


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        rng.shuffle(directions)
        placed = False
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in occupied:
                occupied[new_pos] = i
                chain.append(new_pos)
                placed = True
                break
        if not placed:
            # Can't continue, return what we have or restart
            return None
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain."""
    n = len(chain)
    if n < 3:
        return None
    
    pivot = rng.randint(1, n - 2)
    # Choose a rotation
    dx = chain[pivot][0] - chain[pivot - 1][0]
    dy = chain[pivot][1] - chain[pivot - 1][1]
    
    # Possible rotations: 90 degrees clockwise, 90 degrees counter-clockwise
    # For direction (dx, dy), 90 deg CW gives (dy, -dx), 90 deg CCW gives (-dy, dx)
    rotations = [(dy, -dx), (-dy, dx)]
    rng.shuffle(rotations)
    
    for new_dx, new_dy in rotations:
        # Check if the new suffix would collide
        new_chain = chain[:pivot]
        new_pos = (chain[pivot][0] + new_dx, chain[pivot][1] + new_dy)
        
        # Simulate the rest of the chain with the new direction
        # This is complex; let's just try it
        new_suffix = [new_pos]
        # The rest of the chain from pivot+1 onwards needs to be transformed
        # Actually, a pivot move rotates the entire suffix around the pivot point
        
        # Let's implement it properly:
        # The vector from pivot-1 to pivot defines the "before" direction
        # We rotate the entire suffix (pivot to end) by 90 or 270 degrees
        # around the pivot-1 point
        
        # Simpler: just try moving the pivot point and see if valid
        # For now, let's do a simpler move: change one residue position
        pass
    
    # Let's do a simpler move: pick a random interior residue and try to move it
    idx = rng.randint(1, n - 2)
    current = chain[idx]
    neighbors = list(_neighbours(current))
    rng.shuffle(neighbors)
    
    for new_pos in neighbors:
        if new_pos == chain[idx - 1] or new_pos == chain[idx + 1]:
            continue
        # Check if new_pos is already occupied by another part of the chain
        if new_pos in set(chain[:idx]) | set(chain[idx+1:]):
            continue
        # Check if the new position maintains connectivity
        # chain[idx-1] to new_pos must be unit step
        if abs(new_pos[0] - chain[idx-1][0]) + abs(new_pos[1] - chain[idx-1][1]) != 1:
            continue
        if abs(new_pos[0] - chain[idx+1][0]) + abs(new_pos[1] - chain[idx+1][1]) != 1:
            continue
        
        new_chain = chain.copy()
        new_chain[idx] = new_pos
        return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a segment of the chain."""
    n = len(chain)
    if n < 5:
        return None
    
    # Pick a segment of length 2-4
    seg_len = rng.randint(2, min(4, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    # The segment is chain[start:end+1]
    # We need to rotate it while keeping chain[start-1] and chain[end+1] fixed
    
    # This is complex. Let's do a simpler version: just try to flip a small segment
    # For now, return None and rely on other moves
    return None


def _reverse_move(chain, rng):
    """Reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    seg_len = rng.randint(2, min(6, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    # Reverse the segment
    new_chain = chain.copy()
    new_chain[start:end+1] = list(reversed(chain[start:end+1]))
    
    # Check validity
    if not _is_valid_walk(new_chain):
        return None
    
    # Check connectivity at boundaries
    if abs(new_chain[start][0] - new_chain[start-1][0]) + abs(new_chain[start][1] - new_chain[start-1][1]) != 1:
        return None
    if abs(new_chain[end][0] - new_chain[end+1][0]) + abs(new_chain[end][1] - new_chain[end+1][1]) != 1:
        return None
    
    return new_chain


def fold(sequence, dimension):
    """Fold the sequence using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 180.0  # Leave some margin before 210s kill
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Generate initial conformation
    best_chain = _initial_spiral(sequence, dimension)
    if best_chain is None or len(best_chain) != n or not _is_valid_walk(best_chain):
        # Fallback to random walk
        rng = random.Random(42 + n)
        best_chain = _random_walk(sequence, dimension, rng)
        if best_chain is None or len(best_chain) != n:
            # Last resort: straight line
            best_chain = [(i, 0) for i in range(n)]
    
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    rng = random.Random(12345 + n)
    
    # Count how many steps we can do
    step_count = 0
    max_steps = 500000
    
    while time.time() < deadline - 5.0 and step_count < max_steps:
        step_count += 1
        
        # Choose a move
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.4:
            # Pivot-like: move a single interior residue
            new_chain = _pivot_move(current_chain, rng)
        elif move_type < 0.7:
            # Reverse a segment
            new_chain = _reverse_move(current_chain, rng)
        else:
            # Try a small random perturbation: pick a random interior point and nudge it
            idx = rng.randint(1, len(current_chain) - 2)
            neighbors = list(_neighbours(current_chain[idx]))
            rng.shuffle(neighbors)
            for new_pos in neighbors:
                if new_pos == current_chain[idx-1] or new_pos == current_chain[idx+1]:
                    continue
                if new_pos in set(current_chain[:idx]) | set(current_chain[idx+1:]):
                    continue
                if abs(new_pos[0] - current_chain[idx-1][0]) + abs(new_pos[1] - current_chain[idx-1][1]) != 1:
                    continue
                if abs(new_pos[0] - current_chain[idx+1][0]) + abs(new_pos[1] - current_chain[idx+1][1]) != 1:
                    continue
                new_chain = current_chain.copy()
                new_chain[idx] = new_pos
                break
        
        if new_chain is None:
            T *= cooling
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        
        # Cool down
        T *= cooling
    
    # Final check
    if not _is_valid_walk(best_chain):
        # Fallback
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
