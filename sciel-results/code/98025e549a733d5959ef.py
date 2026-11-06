"""HP lattice protein folding - local search with Monte Carlo moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 295  # safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: negative count of H-H contacts (non-consecutive neighbours)."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, pos, old_pos, new_pos):
    """
    Compute the change in energy when moving residue at position pos from old_pos to new_pos.
    This considers only contacts involving the moved residue for speed.
    """
    n = len(chain)
    # Build position map excluding pos
    where = {}
    for i, p in enumerate(chain):
        if i != pos:
            where[p] = i
    
    # Count contacts involving pos before and after
    def count_contacts(pt, seq_idx):
        if sequence[seq_idx] != "H":
            return 0
        cnt = 0
        for q in _neighbours(pt):
            j = where.get(q)
            if j is not None and j != seq_idx and abs(j - seq_idx) > 1 and sequence[j] == "H":
                cnt += 1
        return cnt
    
    before = count_contacts(old_pos, pos)
    after = count_contacts(new_pos, pos)
    
    return after - before


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot by 90 degrees."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    bond = [chain[pivot + 1][a] - chain[pivot][a] for a in range(dimension)]
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
    new_chain = list(chain[:pivot + 1])
    occupied = set(chain[:pivot + 1])
    
    new_tail = []
    for pt in chain[pivot + 1:]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:  # around x
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:  # around y
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:  # around z
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    new_chain.extend(new_tail)
    return new_chain


def _greedy_init(sequence, dimension, rng):
    """
    Greedy initialization that prefers H-H contacts.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score this site
            score = 0
            if sequence[i] == "H":
                # Check if this site is adjacent to any previous H
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 10
            # Slight preference for compactness
            dist = sum(c * c for c in site)
            score -= dist * 0.1
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate with some randomness
        candidates.sort(key=lambda x: -x[0])
        k = min(3, len(candidates))
        site = rng.choice([c[1] for c in candidates[:k]])
        occupied[site] = i
        chain.append(site)
    
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
            if site in occupied:
                continue
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


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
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


def _local_search(sequence, chain, rng, time_budget, dimension):
    """Monte Carlo local search to improve conformation energy."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0  # initial temperature
    cooling = 0.999
    
    moves_made = 0
    while time.time() - start_time < time_budget:
        moves_made += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot move
            pivot = rng.randint(0, n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        else:
            # Small local move: move one internal residue
            # Pick a random internal position and try to move it
            if n < 3:
                continue
            pos = rng.randint(1, n - 2)
            old_pt = current[pos]
            # Try moving to a neighboring site
            candidates = [q for q in _neighbours(old_pt) if q not in set(current)]
            if not candidates:
                continue
            new_pt = rng.choice(candidates)
            # Check that the move doesn't break adjacency with neighbors
            # Actually, we need to move a segment, not a single point, to maintain unit steps
            # Single point move breaks the chain. Let's do a small pivot instead.
            pivot = pos - 1
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta <= 0:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            # Accept with probability exp(-delta/temp)
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
    
    rng = random.Random(12345 + len(sequence))
    
    # Time budget for this sequence
    # Longer sequences need more time
    base_time = 2.0
    per_residue = 0.15
    total_budget = min(10.0, base_time + per_residue * n)
    
    # Reserve some time
    remaining = _DEADLINE - time.time()
    if remaining < 1.0:
        # Very little time left, just return a valid conformation
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = _helix_init(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Cap the budget
    total_budget = min(total_budget, remaining * 0.8)
    
    best_chain = None
    best_energy = 0
    
    # Generate initial conformations and run local search
    n_initial = max(3, min(10, int(total_budget / 1.5)))
    
    for init_i in range(n_initial):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate initial conformation
        if init_i == 0:
            chain = _random_walk(sequence, dimension, rng)
        elif init_i == 1:
            chain = _helix_init(sequence, dimension, rng)
        elif init_i == 2:
            chain = _greedy_init(sequence, dimension, rng)
        else:
            chain = _random_walk(sequence, dimension, rng)
        
        if chain is None:
            continue
        
        # Verify it's valid
        if len(chain) != n:
            continue
        if len(set(chain)) != n:
            continue
        for i in range(n - 1):
            if sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1])) != 1:
                break
        else:
            # Valid
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = list(chain)
                best_energy = energy
            
            # Run local search with remaining budget
            time_left = _DEADLINE - time.time()
            if time_left < 2.0:
                break
            search_budget = min(2.0, time_left * 0.7 / (n_initial - init_i))
            improved = _local_search(sequence, chain, rng, search_budget, dimension)
            imp_energy = _energy(sequence, improved)
            if imp_energy < best_energy:
                best_chain = improved
                best_energy = imp_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
