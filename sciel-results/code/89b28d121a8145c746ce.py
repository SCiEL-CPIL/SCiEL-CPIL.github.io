"""HP lattice protein folding using Monte Carlo with pivot moves and contact list maintenance."""
# EVOLVE-BLOCK-START
import random
import math
import time


# Precompute direction vectors for 3D
_DIRS3 = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]


def _add3(p, d):
    return (p[0]+d[0], p[1]+d[1], p[2]+d[2])


def _sub3(p, q):
    return (p[0]-q[0], p[1]-q[1], p[2]-q[2])


def _is_adj3(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2]) == 1


def _generate_saw(n, rng, dim=3):
    """Generate a random self-avoiding walk of length n."""
    if n == 1:
        return [(0,0,0)]
    
    for attempt in range(200):
        chain = [(0,0,0)]
        occupied = {(0,0,0)}
        success = True
        for i in range(1, n):
            last = chain[-1]
            neighbors = [_add3(last, d) for d in _DIRS3]
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    
    # Fallback: straight line
    return [(i, 0, 0) for i in range(n)]


def _build_contact_list(sequence, chain):
    """Build a set of (i,j) pairs representing H-H non-consecutive contacts."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for d in _DIRS3:
            q = _add3(p, d)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts.add((i, j))
    return contacts, where


def _pivot_move(chain, pivot_idx, angle, rng, dim=3):
    """
    Perform a pivot move: rotate the sub-chain from pivot_idx to end around pivot.
    Returns new chain or None if invalid.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # Get the tail part to rotate
    tail = chain[pivot_idx:]
    pivot = chain[pivot_idx - 1]
    
    # Compute relative positions from pivot
    rel_tail = [_sub3(p, pivot) for p in tail]
    
    # Apply rotation (simple 90-degree rotations around axes)
    # angle: 0=identity, 1-5 are the 5 non-trivial 90-degree rotations in 3D
    rotated = []
    for p in rel_tail:
        x, y, z = p
        if angle == 0:
            new_p = (x, y, z)
        elif angle == 1:  # rotate 90 deg around x
            new_p = (x, -z, y)
        elif angle == 2:  # rotate 90 deg around y
            new_p = (z, y, -x)
        elif angle == 3:  # rotate 90 deg around z
            new_p = (-y, x, z)
        elif angle == 4:  # rotate 180 deg around x
            new_p = (x, -y, -z)
        else:  # angle == 5: rotate 180 deg around y
            new_p = (-x, y, -z)
        rotated.append(_add3(pivot, new_p))
    
    new_chain = chain[:pivot_idx] + rotated
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    # Check unit steps
    for i in range(1, n):
        if not _is_adj3(new_chain[i], new_chain[i-1]):
            return None
    
    return new_chain


def _compute_energy(sequence, chain):
    """Compute energy as negative number of H-H non-consecutive contacts."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for d in _DIRS3:
            q = _add3(p, d)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _monte_carlo_fold(sequence, dimension, rng, time_limit=8.0):
    """
    Monte Carlo folding with pivot moves.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return _generate_saw(n, rng)
    
    chain = _generate_saw(n, rng)
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 5.0
    min_temp = 0.01
    cooling_rate = 0.99999
    
    # Adaptive parameters
    accepted = 0
    total = 0
    window = 1000
    target_accept = 0.2
    
    steps_since_improve = 0
    max_no_improve = 20000
    
    while time.time() - start_time < time_limit:
        total += 1
        
        # Choose pivot index (bias toward middle for larger moves)
        if n > 10:
            # Weighted random: more likely to pick pivots that create larger segments
            pivot_idx = rng.randint(1, n - 2)
            # Occasionally pick a pivot that creates a very large rotation
            if rng.random() < 0.1:
                pivot_idx = rng.randint(n//3, 2*n//3)
        else:
            pivot_idx = rng.randint(1, n - 2)
        
        # Choose rotation angle (0-5)
        angle = rng.randint(0, 5)
        
        # Try the pivot move
        new_chain = _pivot_move(chain, pivot_idx, angle, rng)
        
        if new_chain is None:
            # Move failed (collision), count as rejected
            continue
        
        # Compute new energy
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e <= 0:
            accept = True
        else:
            # Use exp(-delta_e/temp) but delta_e is positive (worse)
            # Energy is negative, so delta_e > 0 means worse (less negative)
            try:
                prob = math.exp(-delta_e / temp)
            except:
                prob = 0.0
            accept = rng.random() < prob
        
        if accept:
            chain = new_chain
            current_energy = new_energy
            accepted += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        
        else:
            steps_since_improve += 1
        
        # Adaptive temperature
        if total % window == 0:
            accept_rate = accepted / total if total > 0 else 0
            if accept_rate > target_accept * 1.5:
                temp *= 0.8
            elif accept_rate < target_accept * 0.5:
                temp = min(temp * 1.5, 10.0)
            else:
                temp *= cooling_rate
            temp = max(temp, min_temp)
            accepted = 0
            total = 0
        
        # Restart if stuck for too long
        if steps_since_improve > max_no_improve:
            chain = _generate_saw(n, rng)
            current_energy = _compute_energy(sequence, chain)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
            steps_since_improve = 0
            temp = 5.0
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using Monte Carlo with pivot moves.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Use multiple restarts with different seeds
    total_time = 8.0  # seconds per sequence
    num_restarts = 3
    time_per_restart = total_time / num_restarts
    
    best_chain = None
    best_energy = float('inf')
    
    for restart in range(num_restarts):
        seed = 42 + len(sequence) * 1000 + restart * 7
        rng = random.Random(seed)
        
        chain = _monte_carlo_fold(sequence, dimension, rng, time_limit=time_per_restart)
        
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Verify self-avoidance
    seen = set()
    valid = True
    for p in best_chain:
        if p in seen:
            valid = False
            break
        seen.add(p)
    
    # Verify unit steps
    if valid:
        for i in range(1, n):
            diff = abs(best_chain[i][0] - best_chain[i-1][0]) + \
                   abs(best_chain[i][1] - best_chain[i-1][1]) + \
                   abs(best_chain[i][2] - best_chain[i-1][2])
            if diff != 1:
                valid = False
                break
    
    if not valid:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
