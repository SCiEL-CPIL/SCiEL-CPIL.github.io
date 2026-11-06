# EVOLVE-BLOCK-START
import random
import time
import sys

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

def _random_fold(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    Uses a simple backtracking-free heuristic: if stuck, restart.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Try a few times to generate a valid SAW
    for attempt in range(50):
        chain = [(0, 0, 0)]
        occupied = {chain[0]: 0}
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    placed = True
                    break
            if not placed:
                break
        
        if len(chain) == n:
            return chain
    # Fallback: straight line (invalid for H-P contacts but valid geometry)
    return [(i, 0, 0) for i in range(n)]

def _pivot_move(chain, dimension, rng):
    """
    Perform a pivot move on the chain.
    Returns (new_chain, valid)
    """
    n = len(chain)
    if n < 4:
        return None, False
    
    # Choose pivot index. Avoid ends.
    pivot_idx = rng.randint(1, n - 2)
    
    # The head is chain[0..pivot_idx]
    # The tail is chain[pivot_idx+1..n-1]
    # We rotate the tail around the bond (pivot_idx-1, pivot_idx)
    
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    
    # Determine the axis of the bond
    bond_vector = [p_pivot[i] - p_prev[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if bond_vector[i] != 0:
            bond_axis = i
            break
    if bond_axis == -1:
        return None, False
        
    # Choose rotation axis (perpendicular to bond axis)
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    if not rot_axes:
        return None, False
        
    r_axis = rng.choice(rot_axes)
    angle = rng.choice([1, -1]) # 90 or -90 degrees
    
    # Set of head points (indices 0 to pivot_idx)
    head_set = set(chain[:pivot_idx + 1])
    
    # Rotate tail points
    new_tail = []
    valid = True
    occupied_tail = set()
    
    for i in range(pivot_idx + 1, n):
        orig = chain[i]
        
        # Vector from pivot to current point
        dx = orig[0] - p_pivot[0]
        dy = orig[1] - p_pivot[1]
        dz = orig[2] - p_pivot[2]
        
        # Rotate around r_axis
        if r_axis == 0: # Rotate around X
            if angle == 1:
                nx, ny, nz = dx, -dz, dy
            else:
                nx, ny, nz = dx, dz, -dy
        elif r_axis == 1: # Rotate around Y
            if angle == 1:
                nx, ny, nz = dz, dy, -dx
            else:
                nx, ny, nz = -dz, dy, dx
        else: # Rotate around Z
            if angle == 1:
                nx, ny, nz = -dy, dx, dz
            else:
                nx, ny, nz = dy, -dx, dz
            
        new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
        
        if new_point in head_set:
            valid = False
            break
        if new_point in occupied_tail:
            valid = False
            break
            
        occupied_tail.add(new_point)
        new_tail.append(new_point)
        
    if not valid:
        return None, False
        
    new_chain = chain[:pivot_idx + 1] + new_tail
    return new_chain, True

def _kink_move(chain, dimension, rng):
    """
    Perform a kink move (rotation of 3 residues around the axis defined by the 
    vector from residue i-1 to i+2).
    Actually, standard kink move rotates residues i, i+1, i+2 around the axis 
    connecting i-1 and i+3? No, usually it's a local rotation of a 3-residue segment.
    Let's implement a simpler "Crankshaft" or just a local 3-residue rotation.
    For simplicity and robustness, let's implement a rotation of residues i, i+1, i+2
    around the axis defined by the vector (chain[i-1] -> chain[i+3])? That's complex.
    
    Alternative: Rotate residues i, i+1, i+2 around the axis defined by the vector 
    connecting chain[i-1] and chain[i]? No, that's just a pivot with a short tail.
    
    Let's implement a "Kink" as rotating residues i, i+1, i+2 around the axis 
    defined by the vector from chain[i-1] to chain[i+2].
    Wait, the axis must pass through the center of mass or something?
    
    Let's stick to a simpler definition:
    Pick i. Rotate residues i, i+1, i+2 around the axis defined by the vector 
    (chain[i-1] - chain[i+2])?
    
    Actually, a common move is to rotate a segment of length 3 or 4.
    Let's implement a rotation of residues i, i+1, i+2 around the axis 
    defined by the vector connecting chain[i-1] and chain[i+3] (if it exists).
    This is effectively a pivot at i-1 and i+3, which is a "crankshaft" move.
    
    For simplicity in this implementation, I will implement a "Crankshaft" move:
    Rotate residues i, i+1, i+2 around the axis defined by the vector 
    from chain[i-1] to chain[i+3].
    """
    n = len(chain)
    if n < 5:
        return None, False
        
    # i must be such that i-1 >= 0 and i+3 < n
    # So i in [1, n-4]
    i = rng.randint(1, n - 4)
    
    p_start = chain[i-1]
    p_end = chain[i+3]
    
    # Axis vector
    ax = p_end[0] - p_start[0]
    ay = p_end[1] - p_start[1]
    az = p_end[2] - p_start[2]
    
    # Check if axis is valid (non-zero)
    if ax == 0 and ay == 0 and az == 0:
        return None, False
        
    # We need to rotate points p_i, p_{i+1}, p_{i+2} around this axis.
    # This requires 3D rotation math.
    # To keep it simple and avoid bugs, let's just use Pivot moves but 
    # bias the pivot index towards the middle.
    # The inspiration says "Kink moves ... faster local optimization".
    # Let's implement a simple 3-residue rotation around the Z-axis (or any axis) 
    # if the segment is flat? No, that's too restrictive.
    
    # Let's just return None for now and rely on Pivots, or implement a 
    # very simple local move: try to move residue i+1 to a new position 
    # while keeping i and i+2 fixed? That's not a rotation.
    
    # Given the complexity, let's skip Kink moves in this version and rely 
    # on high-frequency Pivot moves with Simulated Annealing, which is 
    # proven to work well (Inspiration 1 & 3).
    return None, False

def _sa_fold(sequence, chain, dimension, rng, deadline):
    """
    Simulated Annealing optimization.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    
    # Initial temperature
    T = 1.0
    # Cooling rate
    alpha = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Choose move
        # 70% Pivot, 30% Kink (but Kink is disabled, so 100% Pivot)
        new_chain, valid = _pivot_move(current_chain, dimension, rng)
        
        if not valid:
            # Try a few more times? Or just break?
            # Let's just continue to next step
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Accept with probability exp(-delta_e / T)
            # delta_e is positive (energy increases, i.e., less negative)
            # So -delta_e is negative
            prob = 2.718281828 ** (-delta_e / T)
            if rng.random() < prob:
                current_chain = new_chain
                current_energy = new_energy
        
        # Cool down
        T *= alpha
        
        # If T is too low, we might get stuck. 
        # We can reheat if no improvement for a while?
        # For now, simple geometric cooling.
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Uses Multi-Start Simulated Annealing with Pivot Moves.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Global time management
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 300.0
    call_deadline = time.time() + 8.0
    if call_deadline > global_deadline:
        call_deadline = global_deadline
    
    n_restarts = 10
    # For long chains, fewer restarts?
    # Let's keep it simple: 10 restarts.
    
    for i in range(n_restarts):
        if time.time() > call_deadline - 1.0:
            break
            
        # Generate initial chain
        chain = _random_fold(sequence, dimension, rng)
        
        # Optimization budget for this restart
        remaining_restarts = n_restarts - i
        time_per_restart = (call_deadline - time.time()) / remaining_restarts
        
        # Run SA
        optimized_chain, opt_energy = _sa_fold(sequence, chain, dimension, rng, time.time() + time_per_restart)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
