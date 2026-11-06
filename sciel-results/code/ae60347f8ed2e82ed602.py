"""HP lattice protein folding - optimized local search with incremental energy."""
import random
import time
import math
import sys

_DEADLINE = time.time() + 315  # safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_map(sequence, chain):
    """Build a dict mapping position -> set of neighbor positions that are H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = {}
    for i in range(n):
        contacts[i] = set()
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts[i].add(j)
                    contacts[j].add(i)
    return contacts


def _energy_from_contacts(contacts):
    """Compute energy from contact map (each contact counted once)."""
    total = 0
    for i, nbrs in contacts.items():
        total += len(nbrs)
    return -total // 2


def _compute_energy(sequence, chain):
    """Compute energy: negative count of H-H contacts (non-consecutive neighbours)."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        if not candidates:
            return None
        site = rng.choice(candidates)
        occupied[site] = i
        chain.append(site)
    return chain


def _spiral_fold(sequence, dimension, rng):
    """Generate a compact spiral-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    # Try to build a compact structure using a spiral pattern
    # Directions in order for a square spiral in the xy-plane
    directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    
    # Spiral pattern: move 1, 1, 2, 2, 3, 3, 4, 4, ...
    step_count = 1
    dir_index = 0
    steps_in_current_dir = 0
    max_steps_in_dir = 1
    
    for i in range(1, n):
        placed = False
        # Try the current spiral direction first
        for attempt in range(dimension * 2):
            if attempt < 4:
                d = directions[attempt % 4]
            else:
                d = (0, 0, (attempt % 2) * 2 - 1)
            
            new_pt = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pt not in occupied:
                chain.append(new_pt)
                occupied[new_pt] = i
                placed = True
                break
        
        if not placed:
            # Dead end in spiral, fall back to random
            candidates = [s for s in _neighbours(chain[-1]) if s not in occupied]
            if not candidates:
                return None
            site = rng.choice(candidates)
            chain.append(site)
            occupied[site] = i
    
    return chain


def _helix_init(sequence, dimension, rng):
    """Generate a compact helical initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                dist = sum(c * c for c in site)
                candidates.append((dist, site))
        if not candidates:
            return None
        candidates.sort()
        k = min(3, len(candidates))
        site = rng.choice([c[1] for c in candidates[:k]])
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot by 90 degrees around an axis."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    bond = [chain[pivot+1][a] - chain[pivot][a] for a in range(dimension)]
    
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    pivot_pt = chain[pivot]
    
    new_tail = []
    for pt in chain[pivot + 1:]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    occupied = set(chain[:pivot + 1])
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    new_chain = list(chain[:pivot + 1])
    new_chain.extend(new_tail)
    return new_chain


def _segment_move(chain, start, end, rng, dimension):
    """Move a segment of the chain (start to end inclusive) to a new location."""
    n = len(chain)
    if start < 1 or end >= n - 1 or end <= start:
        return None
    
    # The segment is chain[start:end+1]
    # It's connected to the rest by bonds at start-1->start and end->end+1
    # We can rotate the segment around one of its endpoints
    
    # Try rotating around the bond (start-1, start)
    bond = [chain[start][a] - chain[start-1][a] for a in range(dimension)]
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    pivot_pt = chain[start - 1]
    
    new_seg = []
    for pt in chain[start:end+1]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_seg.append(new_pt)
    
    # Check self-avoidance: the segment must not overlap with the fixed parts
    fixed = set(chain[:start]) | set(chain[end+1:])
    for pt in new_seg:
        if pt in fixed:
            return None
        fixed.add(pt)
    
    new_chain = list(chain[:start])
    new_chain.extend(new_seg)
    new_chain.extend(chain[end+1:])
    return new_chain


def _local_search(sequence, chain, rng, time_budget, dimension):
    """Monte Carlo local search with incremental energy tracking."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 2.0
    cooling = 0.9995
    
    # Track contacts for incremental updates
    where = {p: i for i, p in enumerate(current)}
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    while time.time() - start_time < time_budget:
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot move
            pivot = rng.randint(0, n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        elif move_type < 0.8:
            # Segment move
            if n < 4:
                continue
            seg_len = min(rng.randint(1, min(10, n // 3)), n - 2)
            start = rng.randint(1, n - 1 - seg_len)
            end = start + seg_len
            new_chain = _segment_move(current, start, end, rng, dimension)
            if new_chain is None:
                continue
        else:
            # Small pivot near the ends
            if rng.random() < 0.5:
                pivot = rng.randint(0, min(5, n - 2))
            else:
                pivot = rng.randint(max(0, n - 6), n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            if temp > 0.01:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_chain
                    current_energy = new_energy
        
        temp *= cooling
    
    return best


def fold(sequence, dimension):
    """Best conformation using local search from multiple initial conformations."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence) * 7)
    
    # Time budget for this sequence
    remaining = _DEADLINE - time.time()
    if remaining < 1.0:
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = _spiral_fold(sequence, dimension, rng)
        if chain is None:
            chain = _helix_init(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Allocate time based on sequence length
    # Shorter sequences can afford more restarts
    if n <= 48:
        n_initial = 8
        per_search_time = 3.0
    elif n <= 80:
        n_initial = 5
        per_search_time = 4.0
    else:
        n_initial = 3
        per_search_time = 5.0
    
    # Cap total time
    total_budget = min(remaining * 0.9, n_initial * per_search_time)
    
    best_chain = None
    best_energy = 0
    
    for init_i in range(n_initial):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate initial conformation
        if init_i == 0:
            chain = _spiral_fold(sequence, dimension, rng)
        elif init_i == 1:
            chain = _helix_init(sequence, dimension, rng)
        elif init_i == 2:
            chain = _random_walk(sequence, dimension, rng)
        else:
            # Mix of strategies
            if init_i % 2 == 0:
                chain = _spiral_fold(sequence, dimension, rng)
            else:
                chain = _random_walk(sequence, dimension, rng)
        
        if chain is None:
            chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
        # Verify validity
        if len(chain) != n:
            continue
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1])) != 1:
                valid = False
                break
        if not valid:
            continue
        
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # Run local search
        time_left = _DEADLINE - time.time()
        if time_left < 2.0:
            break
        search_budget = min(per_search_time, time_left * 0.8 / max(1, n_initial - init_i))
        improved = _local_search(sequence, chain, rng, search_budget, dimension)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_chain = improved
            best_energy = imp_energy
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
