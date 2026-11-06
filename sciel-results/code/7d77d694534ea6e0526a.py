"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 310  # safety margin before 330s kill


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


def _pivot_move(chain, rng, dimension):
    """Perform a pivot move: rotate a subchain by 90 degrees around the bond axis."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose a pivot point (not at ends)
    pivot = rng.randint(1, n - 2)
    
    # Get the bond direction
    bond = tuple(chain[pivot][a] - chain[pivot - 1][a] for a in range(dimension))
    
    # Find the axis of the bond
    axis = None
    for a in range(dimension):
        if bond[a] != 0:
            axis = a
            break
    if axis is None:
        return None
    
    # The other two axes
    other_axes = [a for a in range(dimension) if a != axis]
    
    # Choose a rotation direction (90, -90, 180)
    angle = rng.choice([1, -1, 2])  # 1=90, -1=-90, 2=180
    
    # Determine the rotation
    # We rotate around the bond axis
    # For 90 degree rotation, we swap the two other axes with signs
    # For 180, we negate both other axes
    
    # Build the rotation matrix for the subchain
    def rotate_point(point, pivot_point):
        # Translate so pivot is at origin
        rel = tuple(point[a] - pivot_point[a] for a in range(dimension))
        
        if angle == 1:  # 90 degrees
            # Rotate in the plane of the two other axes
            new_rel = list(rel)
            new_rel[other_axes[0]] = rel[other_axes[1]]
            new_rel[other_axes[1]] = -rel[other_axes[0]]
            rel = tuple(new_rel)
        elif angle == -1:  # -90 degrees
            new_rel = list(rel)
            new_rel[other_axes[0]] = -rel[other_axes[1]]
            new_rel[other_axes[1]] = rel[other_axes[0]]
            rel = tuple(new_rel)
        elif angle == 2:  # 180 degrees
            new_rel = list(rel)
            new_rel[other_axes[0]] = -rel[other_axes[0]]
            new_rel[other_axes[1]] = -rel[other_axes[1]]
            rel = tuple(new_rel)
        
        # Translate back
        return tuple(rel[a] + pivot_point[a] for a in range(dimension))
    
    # Apply the rotation to the subchain
    subchain = chain[pivot:]
    new_subchain = [rotate_point(p, chain[pivot]) for p in subchain]
    
    # Check if the move is valid (self-avoiding)
    occupied = set(chain[:pivot])
    for i, p in enumerate(new_subchain):
        if p in occupied:
            return None
        occupied.add(p)
    
    # Also check that the bond from pivot-1 to pivot is still valid
    # (it should be, since we're rotating around that bond)
    
    new_chain = chain[:pivot] + new_subchain
    return new_chain


def _local_move(chain, rng, dimension, sequence):
    """Perform a local move: try to move a single residue to an adjacent empty site."""
    n = len(chain)
    if n < 2:
        return None
    
    # Choose a random internal residue (not at ends)
    i = rng.randint(1, n - 2)
    
    occupied = set(chain)
    occupied.discard(chain[i])
    
    # Try all 6 neighbors
    for site in _neighbours(chain[i]):
        if site in occupied:
            continue
        # Check that the new position doesn't break the chain
        # The residue at i must still be connected to i-1 and i+1
        # Since we're moving to an adjacent site, this should be fine
        # as long as the new site is adjacent to both neighbors
        # Actually, we need to ensure the chain remains connected
        # The new site must be adjacent to chain[i-1] and chain[i+1]
        # But that's not necessarily true, so this move might break the chain
        # Let's just try it and check self-avoidance
        new_chain = chain[:]
        new_chain[i] = site
        # Check self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        # Check that consecutive residues are still adjacent
        valid = True
        for j in range(len(new_chain) - 1):
            diff = sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension))
            if diff != 1:
                valid = False
                break
        if not valid:
            continue
        return new_chain
    
    return None


def _anneal(sequence, chain, dimension, rng, time_limit):
    """Simulated annealing to improve the chain."""
    n = len(sequence)
    if n < 4:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    max_time = min(time_limit, 300)
    
    # Cooling schedule
    T_start = 5.0
    T_end = 0.01
    alpha = 0.995  # cooling rate
    
    T = T_start
    steps = 0
    
    while T > T_end:
        if time.time() - start_time > max_time:
            break
        
        # Choose a move
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, rng, dimension)
        else:
            new_chain = _local_move(chain, rng, dimension, sequence)
        
        if new_chain is None:
            T *= alpha
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = chain[:]
                best_energy = current_energy
        
        T *= alpha
        steps += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using randomized growth and simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Generate multiple initial solutions and anneal each
    num_initial = min(20, max(5, 400 // n))
    
    for trial in range(num_initial):
        if time.time() > _DEADLINE:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Anneal this chain
        time_left = _DEADLINE - time.time()
        if time_left < 1:
            # Just evaluate without annealing
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
            continue
        
        # Allocate time for annealing
        anneal_time = min(10, time_left * 0.8)
        chain = _anneal(sequence, chain, dimension, rng, anneal_time)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
