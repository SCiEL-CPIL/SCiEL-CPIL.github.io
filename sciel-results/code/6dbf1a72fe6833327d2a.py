"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time management
_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 180.0  # Safety margin for 210s kill

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calculate_energy(sequence, chain):
    """Calculate the energy of a conformation. Lower is better (more negative)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk (SARW) to initialize the chain."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a segment of the chain."""
    n = len(chain)
    if n < 3:
        return None
    # Pick a pivot point
    pivot_idx = rng.randint(1, n - 2)
    pivot_point = chain[pivot_idx]
    # Pick a new direction for the segment after the pivot
    # The segment [pivot_idx+1 ... n-1] will be rotated/translated
    # We need to pick a new neighbor for pivot_idx+1 that is not occupied by [0...pivot_idx]
    
    # Occupied set for the "root" part [0...pivot_idx]
    root_occupied = set(chain[:pivot_idx+1])
    
    # The "tail" is chain[pivot_idx+1:]
    tail = chain[pivot_idx+1:]
    
    # We want to move the tail so that its first element is a neighbor of pivot_point
    # and the rest of the tail is valid relative to itself and the root.
    # This is tricky to do efficiently without checking the whole tail each time.
    # A simpler approach for 2D: try all 4 neighbors of pivot_point.
    
    for new_first in _neighbours(pivot_point):
        if new_first in root_occupied:
            continue
        
        # Try to place the tail starting at new_first
        # We need to map the relative coordinates of the tail to the new start
        # Actually, a pivot move rotates the tail around the pivot.
        # In 2D, there are 3 possible rotations (90, 180, 270) or reflections.
        # Let's just try to translate the tail such that chain[pivot_idx+1] becomes new_first.
        # This is a "translation" of the tail, which is a valid move if it doesn't clash.
        
        dx = new_first[0] - chain[pivot_idx+1][0]
        dy = new_first[1] - chain[pivot_idx+1][1]
        
        new_tail = []
        valid = True
        # Check if the translated tail clashes with root or itself
        # We also need to check if it clashes with the "virtual" position of the pivot? No, pivot is fixed.
        # The new tail must not intersect root_occupied.
        # The new tail must be self-avoiding (it is, by translation).
        
        # Optimization: Check against root_occupied
        for p in tail:
            np = (p[0] + dx, p[1] + dy)
            if np in root_occupied:
                valid = False
                break
            new_tail.append(np)
        
        if valid:
            # Construct new chain
            return chain[:pivot_idx+1] + new_tail
    return None

def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a subsegment."""
    n = len(chain)
    if n < 5:
        return None
    
    # Pick a subsegment [i, j] to rotate
    # Length of subsegment must be at least 2
    # Let's pick a random length 2 to 5
    length = rng.randint(2, min(5, n - 2))
    start_idx = rng.randint(1, n - length - 1)
    
    # The subsegment is chain[start_idx : start_idx+length]
    # It is connected to chain[start_idx-1] and chain[start_idx+length]
    p_prev = chain[start_idx - 1]
    p_next = chain[start_idx + length]
    
    # We want to rotate the subsegment around p_prev or p_next?
    # Standard crankshaft: rotate the subsegment such that its endpoints stay connected to the rest?
    # Actually, a crankshaft move moves a segment of length L (2 <= L <= 4 usually) 
    # by rotating it around one of its endpoints or shifting it.
    # Let's use a simpler "subsegment rotation" where we rotate the subsegment 
    # around its first point (p_start = chain[start_idx]).
    
    p_start = chain[start_idx]
    subseg = chain[start_idx : start_idx+length]
    
    # The rest of the chain is:
    # head = chain[:start_idx]
    # tail = chain[start_idx+length:]
    
    # We need to find a new position for the subsegment such that:
    # 1. subseg[0] is still p_start (fixed)
    # 2. The new subseg is self-avoiding
    # 3. The new subseg does not intersect head
    # 4. The new subseg[ -1 ] (last point) must be a neighbor of p_next? 
    #    Wait, if we rotate around p_start, the last point moves. 
    #    The connection to p_next is broken unless the new last point is p_next.
    #    This is a "hinge" move.
    
    # Let's try all 4 possible directions for the first step of the subsegment from p_start.
    # The subsegment is a path of length 'length'.
    # We can generate all possible paths of length 'length' starting at p_start? 
    # That's 3^(length-1) possibilities. For length 4, 27 possibilities. Feasible.
    
    head_occupied = set(chain[:start_idx])
    
    def generate_paths(start, length, occupied, rng):
        """Generate random self-avoiding paths of given length from start."""
        paths = []
        current = [start]
        current_occupied = {start}
        
        def backtrack(path, occ):
            if len(path) == length + 1:
                # Check if the last point is a neighbor of p_next? 
                # No, for a valid move, the new last point must be connected to p_next.
                # So path[-1] must be a neighbor of p_next.
                if _is_neighbor(path[-1], p_next):
                    # Check if path intersects head_occupied
                    for p in path:
                        if p in head_occupied:
                            return
                    paths.append(path)
                return
            
            for nb in _neighbours(path[-1]):
                if nb not in occ and nb not in head_occupied:
                    path.append(nb)
                    occ.add(nb)
                    backtrack(path, occ)
                    path.pop()
                    occ.remove(nb)
        
        backtrack(current, current_occupied)
        return paths
    
    # This might be slow if we do it for every move. 
    # Let's just try a few random rotations.
    # In 2D, we can rotate the subsegment by 90, 180, 270 degrees around p_start.
    
    def rotate_point(p, center, angle):
        # angle in radians
        x, y = p[0] - center[0], p[1] - center[1]
        ca, sa = math.cos(angle), math.sin(angle)
        nx, ny = ca*x - sa*y, sa*x + ca*y
        return (int(round(center[0] + nx)), int(round(center[1] + ny)))
    
    for angle in [math.pi/2, math.pi, 3*math.pi/2]:
        new_subseg = [rotate_point(p, p_start, angle) for p in subseg]
        
        # Check validity
        valid = True
        for p in new_subseg:
            if p in head_occupied:
                valid = False
                break
        
        if valid and _is_neighbor(new_subseg[-1], p_next):
            # Also check self-avoidance of new_subseg (it should be, as rotation is isometry)
            # But we must ensure it doesn't intersect itself? Rotation of a SAW is a SAW.
            return chain[:start_idx] + new_subseg + chain[start_idx+length:]
            
    return None

def _is_neighbor(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1

def _shift_move(chain, rng):
    """Move a single residue to a neighboring empty site."""
    n = len(chain)
    if n < 2:
        return None
    i = rng.randint(1, n - 2) # Don't move ends
    p = chain[i]
    occupied = set(chain)
    for nb in _neighbours(p):
        if nb not in occupied:
            # Check if moving i to nb is valid
            # The neighbors of i in the chain are i-1 and i+1.
            # If nb is adjacent to i-1 or i+1, it creates a "kink" which is fine, 
            # but we must ensure it doesn't clash with other parts.
            # Since nb is not in occupied, it's free.
            # We just need to make sure the new position doesn't clash with i-1 or i+1? 
            # No, i-1 and i+1 are already in occupied. 
            # The only constraint is self-avoidance.
            # Since nb is not in occupied, and we are removing p and adding nb,
            # the only potential clash is if nb is adjacent to something it shouldn't be? 
            # No, adjacency is allowed. Clashing (same position) is not.
            # So if nb is not in occupied, it's a valid move.
            new_chain = chain[:]
            new_chain[i] = nb
            return new_chain
    return None

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # 1. Initialize
    # Try a few random walks to get a good start
    best_chain = None
    best_energy = 0
    for _ in range(50):
        chain = _random_walk(sequence, dimension, rng)
        if chain:
            e = _calculate_energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    
    if best_chain is None:
        # Fallback: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    current_chain = best_chain
    current_energy = best_energy
    
    # 2. Simulated Annealing
    start_time = time.time()
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 200
    
    while temp > min_temp:
        if _time_left() < 5.0:
            break
        
        # Generate a move
        move_type = rng.random()
        new_chain = None
        if move_type < 0.5:
            new_chain = _pivot_move(current_chain, rng)
        elif move_type < 0.8:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _shift_move(current_chain, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _calculate_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = current_chain
                best_energy = current_energy
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart from best found so far with higher temp?
            # Or just cool down faster
            temp *= 0.95
            no_improve_count = 0
        else:
            temp *= cooling_rate
            
    # 3. Local Search (Hill Climbing) on the best found
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        for i in range(len(best_chain)):
            if _time_left() < 2.0:
                break
            p = best_chain[i]
            for nb in _neighbours(p):
                occupied = set(best_chain)
                if nb in occupied:
                    continue
                new_chain = best_chain[:]
                new_chain[i] = nb
                new_energy = _calculate_energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True
                    break # Restart neighborhood search for this i? Or just continue
                    # Actually, if we improve, we should restart the scan from i=0
                    # to ensure we don't miss better moves.
                    break
            if improved:
                break
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
