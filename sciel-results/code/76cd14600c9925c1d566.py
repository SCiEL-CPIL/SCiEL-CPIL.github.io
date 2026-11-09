"""HP lattice protein folding: simulated annealing with chain moves on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 205.0  # safety margin before 210s kill
_START_TIME = None


def _remaining_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_self_avoiding(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum((a - b) ** 2 for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _compute_contacts(sequence, chain):
    """Return a set of contact pairs (i, j) where i < j, both H, not adjacent in chain, and neighbours on lattice."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts


def _energy(sequence, chain):
    """Energy = minus the number of H-H contacts."""
    return -len(_compute_contacts(sequence, chain))


def _initial_chain(sequence, dimension):
    """Create an initial self-avoiding walk (straight line)."""
    n = len(sequence)
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk of the given length."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0,) * dimension]
    occupied = {chain[0]}
    for _ in range(n - 1):
        last = chain[-1]
        options = [site for site in _neighbours(last) if site not in occupied]
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    return chain


def _move_segment(chain, start, end, direction, occupied_set):
    """
    Try to move a segment of the chain (from index start to end inclusive) by direction.
    direction is a tuple like (1,0), (0,1), etc.
    Returns (new_chain, new_occupied) if successful, else (None, None).
    """
    n = len(chain)
    # Check that the segment can be moved without self-collision
    # The segment is chain[start:end+1]
    # After moving, check:
    # 1. Each moved position is not occupied by a non-moved part
    # 2. The moved segment itself remains self-avoiding
    # 3. The connections at the boundaries are valid (unit steps)
    
    moved_indices = set(range(start, end + 1))
    
    # Check boundary connections
    # Left boundary: if start > 0, the new position of chain[start] must be adjacent to chain[start-1]
    if start > 0:
        new_start_pos = tuple(a + b for a, b in zip(chain[start], direction))
        prev_pos = chain[start - 1]
        dist = sum((a - b) ** 2 for a, b in zip(new_start_pos, prev_pos))
        if dist != 1:
            return None, None
    
    # Right boundary: if end < n-1, the new position of chain[end] must be adjacent to chain[end+1]
    if end < n - 1:
        new_end_pos = tuple(a + b for a, b in zip(chain[end], direction))
        next_pos = chain[end + 1]
        dist = sum((a - b) ** 2 for a, b in zip(new_end_pos, next_pos))
        if dist != 1:
            return None, None
    
    # Build new chain and check for collisions
    new_chain = list(chain)
    new_occupied = set()
    
    # First, mark all non-moved positions
    for i in range(n):
        if i not in moved_indices:
            new_occupied.add(chain[i])
    
    # Now place the moved segment
    for i in range(start, end + 1):
        new_pos = tuple(a + b for a, b in zip(chain[i], direction))
        if new_pos in new_occupied:
            return None, None
        new_occupied.add(new_pos)
        new_chain[i] = new_pos
    
    return new_chain, new_occupied


def _single_move(chain, index, direction, occupied_set):
    """
    Try to move a single residue at index by direction.
    Returns (new_chain, new_occupied) if successful, else (None, None).
    """
    n = len(chain)
    if n == 0:
        return None, None
    
    new_chain = list(chain)
    new_occupied = set(chain)
    
    # Remove old position
    old_pos = chain[index]
    new_occupied.discard(old_pos)
    
    # Compute new position
    new_pos = tuple(a + b for a, b in zip(old_pos, direction))
    
    # Check if new position is free
    if new_pos in new_occupied:
        return None, None
    
    # Check boundary connections
    if index > 0:
        prev_pos = chain[index - 1]
        dist = sum((a - b) ** 2 for a, b in zip(new_pos, prev_pos))
        if dist != 1:
            return None, None
    
    if index < n - 1:
        next_pos = chain[index + 1]
        dist = sum((a - b) ** 2 for a, b in zip(new_pos, next_pos))
        if dist != 1:
            return None, None
    
    new_chain[index] = new_pos
    new_occupied.add(new_pos)
    
    return new_chain, new_occupied


def _simulated_annealing(sequence, dimension, rng, time_budget):
    """
    Perform simulated annealing to find a low-energy conformation.
    Returns the best chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start with a random walk or straight line
    chain = _random_walk(sequence, dimension, rng)
    if chain is None:
        chain = _initial_chain(sequence, dimension)
    
    occupied = set(chain)
    current_contacts = _compute_contacts(sequence, chain)
    current_energy = -len(current_contacts)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T0 = 1.0
    T_end = 0.01
    total_time = time_budget
    start_time = time.time()
    
    while _remaining_time() > 1.0:
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T0 * (T_end / T0) ** progress
        
        # Choose a move
        move_type = rng.random()
        if move_type < 0.3:
            # Single residue move
            index = rng.randint(0, n - 1)
            direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            new_chain, new_occupied = _single_move(chain, index, direction, occupied)
        elif move_type < 0.7:
            # Segment move (short)
            length = rng.randint(1, min(5, n))
            start = rng.randint(0, n - length)
            end = start + length - 1
            direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            new_chain, new_occupied = _move_segment(chain, start, end, direction, occupied)
        else:
            # Segment move (medium)
            length = rng.randint(1, min(15, n))
            start = rng.randint(0, n - length)
            end = start + length - 1
            direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            new_chain, new_occupied = _move_segment(chain, start, end, direction, occupied)
        
        if new_chain is None:
            continue
        
        # Compute new contacts and energy
        new_contacts = _compute_contacts(sequence, new_chain)
        new_energy = -len(new_contacts)
        
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            occupied = new_occupied
            current_contacts = new_contacts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy self-avoiding conformation.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time based on sequence length
    # Shorter sequences get less time, longer get more
    # Total budget is about 200s for all sequences
    time_budget = min(20.0, 2.0 + 0.3 * n)
    
    best_chain = None
    best_energy = 1
    
    # Try multiple restarts
    remaining = time_budget
    restart_count = 0
    
    while remaining > 0.5 and _remaining_time() > 1.0:
        restart_count += 1
        chain = _simulated_annealing(sequence, dimension, rng, remaining)
        
        if chain is not None:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
        
        # Reduce remaining time for next restart
        remaining *= 0.5
        if remaining < 1.0:
            break
    
    if best_chain is None:
        best_chain = _initial_chain(sequence, dimension)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
