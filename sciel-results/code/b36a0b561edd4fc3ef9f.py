import random
import time
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 230.0  # Safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, noise=0.1):
    """Greedy chain growth with some randomness."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + noise * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _make_spiral_seed(sequence, dimension):
    """Create a dense spiral-like initial conformation."""
    n = len(sequence)
    chain = []
    occupied = set()
    
    # Try to make a compact spiral in 2D plane (z=0)
    # Simple approach: grow in a pattern that fills space
    x, y, z = 0, 0, 0
    chain.append((x, y, z))
    occupied.add((x, y, z))
    
    # Directions in order for a spiral-like growth
    dx, dy = 1, 0
    steps_in_direction = 0
    max_steps = 2
    
    for i in range(1, n):
        placed = False
        for _ in range(12):  # Try up to 12 direction changes
            nx, ny, nz = x + dx, y + dy, z
            if (nx, ny, nz) not in occupied:
                chain.append((nx, ny, nz))
                occupied.add((nx, ny, nz))
                x, y, z = nx, ny, nz
                steps_in_direction += 1
                placed = True
                break
            # Rotate direction
            if dx == 1:
                dx, dy = 0, 1
            elif dy == 1:
                dx, dy = -1, 0
            elif dx == -1:
                dx, dy = 0, -1
            elif dy == -1:
                dx, dy = 1, 0
            steps_in_direction = 0
        
        if not placed:
            # Fallback: try all neighbours
            for site in _neighbours((x, y, z)):
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    x, y, z = site
                    placed = True
                    break
        
        if not placed:
            return None
    
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a segment of the chain."""
    n = len(chain)
    if n < 3:
        return None
    
    # Choose pivot point
    pivot = rng.randint(1, n - 1)
    
    # Choose rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.choice([90, -90, 180])
    
    # Rotation matrices for 90, -90, 180 degrees
    def rotate(point, axis, angle):
        p = list(point)
        if angle == 90:
            if axis == 0:
                p[1], p[2] = -p[2], p[1]
            elif axis == 1:
                p[0], p[2] = p[2], -p[0]
            else:
                p[0], p[1] = -p[1], p[0]
        elif angle == -90:
            if axis == 0:
                p[1], p[2] = p[2], -p[1]
            elif axis == 1:
                p[0], p[2] = -p[2], p[0]
            else:
                p[0], p[1] = p[1], -p[0]
        else:  # 180
            if axis == 0:
                p[1], p[2] = -p[1], -p[2]
            elif axis == 1:
                p[0], p[2] = -p[0], -p[2]
            else:
                p[0], p[1] = -p[0], -p[1]
        return tuple(p)
    
    # Apply rotation to segment [pivot, n)
    new_chain = list(chain[:pivot])
    for i in range(pivot, n):
        # Translate to origin, rotate, translate back
        orig = chain[pivot]
        rel = tuple(chain[i][a] - orig[a] for a in range(3))
        rot = rotate(rel, axis, angle)
        new_chain.append(tuple(orig[a] + rot[a] for a in range(3)))
    
    # Check self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
    
    # Check unit steps
    for i in range(1, len(new_chain)):
        diff = sum(abs(new_chain[i][a] - new_chain[i-1][a]) for a in range(3))
        if diff != 1:
            return None
    
    return new_chain


def _slide_move(chain, rng):
    """Perform a slide move: move a single residue to an adjacent empty site."""
    n = len(chain)
    if n < 2:
        return None
    
    # Choose a residue to move (not endpoints)
    i = rng.randint(1, n - 2)
    
    # Try to move to a neighbour
    for site in _neighbours(chain[i]):
        if site in chain:
            continue
        # Check if moving is valid (doesn't break bonds)
        new_chain = list(chain)
        new_chain[i] = site
        # Check unit steps with neighbours
        diff_prev = sum(abs(site[a] - new_chain[i-1][a]) for a in range(3))
        diff_next = sum(abs(new_chain[i+1][a] - site[a]) for a in range(3))
        if diff_prev == 1 and diff_next == 1:
            return new_chain
    
    return None


def _regrow_tail(sequence, chain, split_idx, rng):
    """Regrow the tail of the chain from split_idx onwards."""
    n = len(sequence)
    if split_idx >= n - 1:
        return None
    
    prefix = chain[:split_idx + 1]
    occupied = {p: i for i, p in enumerate(prefix)}
    current = prefix[-1]
    new_chain = list(prefix)
    
    for i in range(split_idx + 1, n):
        options = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.3
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score, best_site = max(options)
        new_chain.append(best_site)
        occupied[best_site] = i
        current = best_site
    
    return new_chain


def _local_search(sequence, chain, time_budget, rng):
    """Local search using pivot and slide moves."""
    if chain is None:
        return None, 1
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Choose move type
        if rng.random() < 0.5:
            new_chain = _pivot_move(current, rng)
        else:
            new_chain = _slide_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        T *= cooling
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using multi-start simulated annealing with regrowth."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Calculate time budget for this sequence
    elapsed = time.time() - _START_TIME
    remaining = max(0, _TOTAL_BUDGET - elapsed)
    
    if remaining < 1.0:
        # Not enough time, return a simple conformation
        return [[i, 0, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 1  # Start with a high energy (worst case)
    
    # Generate diverse initial conformations
    n_restarts = min(8, max(1, int(remaining / 6.0)))
    time_per_restart = remaining / n_restarts
    
    for restart in range(n_restarts):
        # Generate initial conformation
        if restart == 0:
            init = _make_spiral_seed(sequence, dimension)
        else:
            init = _greedy_grow(sequence, dimension, rng, noise=0.1 + 0.1 * restart)
        
        if init is None:
            init = _make_spiral_seed(sequence, dimension)
        
        if init is None:
            # Fallback: straight line
            init = [(i, 0, 0) for i in range(n)]
        
        # Run local search
        result, energy = _local_search(sequence, init, time_per_restart * 0.85, rng)
        
        if result is not None and energy < best_energy:
            best_chain = result
            best_energy = energy
        
        # Check if we're running out of time
        if time.time() - _START_TIME > _TOTAL_BUDGET - 2.0:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
