import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    # chain is list of tuples
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _compact_initial_chain(sequence, dimension, rng):
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {chain[0]}
    x, y, z = 0, 0, 0
    # Simple spiral growth
    # Directions: +x, +y, +z, -x, -y, -z ... 
    # To keep it compact, we can use a space-filling curve heuristic or just random walk with bias to center.
    # For now, a simple random walk with bias to stay near origin is okay for initialization.
    for i in range(1, n):
        placed = False
        # Try directions in random order, biased towards unoccupied neighbors that are closer to origin
        dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        rng.shuffle(dirs)
        for dx, dy, dz in dirs:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) not in occupied:
                x, y, z = nx, ny, nz
                chain.append((x, y, z))
                occupied.add((x, y, z))
                placed = True
                break
        if not placed:
            # If stuck, just break, we will fix in SA
            break
    return chain

def _pivot_move(chain, i, angle, rng):
    """
    Perform a pivot move: rotate chain[i:] around the bond chain[i-1]-chain[i].
    angle: 90, 180, 270.
    Returns new chain (list of tuples) or None if invalid (self-intersection).
    """
    if i <= 0 or i >= len(chain):
        return None
    
    p0 = chain[i-1]
    p1 = chain[i]
    
    # Determine axis
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    
    # Find the axis (0 for x, 1 for y, 2 for z)
    axis = -1
    if abs(dx) == 1: axis = 0
    elif abs(dy) == 1: axis = 1
    elif abs(dz) == 1: axis = 2
    
    if axis == -1:
        return None # Invalid bond
        
    new_chain = list(chain)
    occupied = set(chain[:i]) # Points before i are fixed
    
    # Rotate points from i to end
    for k in range(i, len(chain)):
        px, py, pz = chain[k]
        # Relative to p0
        rx = px - p0[0]
        ry = py - p0[1]
        rz = pz - p0[2]
        
        # Rotate (rx, ry, rz) by angle around axis
        if axis == 0: # X-axis
            if angle == 90:
                nx, ny, nz = rx, -rz, ry
            elif angle == 180:
                nx, ny, nz = rx, -ry, -rz
            elif angle == 270:
                nx, ny, nz = rx, rz, -ry
            else:
                nx, ny, nz = rx, ry, rz
        elif axis == 1: # Y-axis
            if angle == 90:
                nx, ny, nz = rz, ry, -rx
            elif angle == 180:
                nx, ny, nz = -rx, ry, -rz
            elif angle == 270:
                nx, ny, nz = -rz, ry, rx
            else:
                nx, ny, nz = rx, ry, rz
        elif axis == 2: # Z-axis
            if angle == 90:
                nx, ny, nz = ry, -rx, rz
            elif angle == 180:
                nx, ny, nz = -rx, -ry, rz
            elif angle == 270:
                nx, ny, nz = -ry, rx, rz
            else:
                nx, ny, nz = rx, ry, rz
                
        new_p = (p0[0] + nx, p0[1] + ny, p0[2] + nz)
        
        if new_p in occupied:
            return None # Collision
        occupied.add(new_p)
        new_chain[k] = new_p
        
    return new_chain

def _sa_fold(sequence, dimension, rng, time_budget):
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = _compact_initial_chain(sequence, dimension, rng)
    if len(chain) < n:
        # Fallback: extend with random walk if possible, or just accept shorter? 
        # No, must be length n.
        # If compact init failed, use a straight line? 
        # Straight line is valid but high energy.
        chain = [(i, 0, 0) for i in range(n)]
        
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    t_start = time.time()
    t_max = time_budget
    T = max(1.0, abs(current_energy) * 0.5)
    cooling_rate = 0.999
    t_min = 0.01
    
    while T > t_min:
        if time.time() - t_start > t_max:
            break
            
        # Pick pivot index i (1 to n-1)
        i = rng.randint(1, n - 1)
        # Pick angle
        angle = rng.choice([90, 180, 270])
        
        new_chain = _pivot_move(chain, i, angle, rng)
        
        if new_chain is None:
            T *= cooling_rate
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    global_deadline = time.time() + 290
    
    # Adjust replicas based on length
    if n <= 48:
        num_replicas = 10
    elif n <= 80:
        num_replicas = 6
    else:
        num_replicas = 4
        
    per_replica_time = 280.0 / num_replicas
    
    best_chain = None
    best_energy = 0
    
    base_seed = 12345 + len(sequence)
    
    for rep in range(num_replicas):
        if time.time() > global_deadline - 5:
            break
        rng = random.Random(base_seed + rep * 7919)
        chain_tuples = _sa_fold(sequence, dimension, rng, per_replica_time)
        # chain_tuples is list of tuples
        energy = _energy(sequence, chain_tuples)
        if best_chain is None or energy < best_energy:
            best_chain = chain_tuples
            best_energy = energy
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Convert to list of lists of ints for return
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
