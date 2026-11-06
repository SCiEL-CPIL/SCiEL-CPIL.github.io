import random
import time
import math

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _calc_energy(sequence, chain):
    """Calculate total energy of a chain."""
    n = len(chain)
    if n < 2:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    is_h = [1 if s == 'H' else 0 for s in sequence]
    energy = 0
    for i in range(n):
        if is_h[i]:
            p = chain[i]
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    energy -= 1
    return energy

def _generate_initial_chain(sequence, dimension, rng):
    """
    Generate a valid self-avoiding walk quickly.
    We try to create a compact random walk.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start with a straight line, then try to "crumple" it using random valid moves?
    # Or just do a random walk with backtracking.
    # A simple random walk often gets stuck. 
    # Let's use a "biased random walk" that prefers unvisited neighbors.
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last_pos = chain[-1]
        neighbors = list(_neighbours(last_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied[nb] = i
                moved = True
                break
        
        if not moved:
            # If stuck, we can't easily backtrack in this simple loop.
            # Fallback: extend in a direction that is free. 
            # If all 6 are occupied, we are in a cage. 
            # For safety, let's just break and return what we have? No, must be len n.
            # Let's try a more robust init: Straight line is always valid.
            # To get some energy, we can do a "snake" pattern.
            # Let's just use a straight line for guaranteed validity and speed, 
            # and rely on SA to fold it.
            # A straight line has 0 energy, but SA will improve it.
            # However, starting from a straight line is a very bad start for SA.
            # Let's try a simple helical/snake pattern which is valid and compact.
            
            # Snake pattern:
            # (0,0,0) -> (1,0,0) -> (1,1,0) -> (0,1,0) -> (0,2,0) ...
            # This is valid.
            # Let's just generate a snake.
            chain = []
            occupied = set()
            x, y, z = 0, 0, 0
            chain.append((x,y,z))
            occupied.add((x,y,z))
            
            # Directions for a simple space-filling curve (approx)
            # Just zigzag on XY plane for simplicity, then extend Z if needed?
            # Actually, let's just use a straight line. It's the safest.
            # The SA will do the heavy lifting.
            # To make SA faster, we can initialize with a random valid walk using a proper backtracking DFS.
            pass

    # The above logic is getting complicated. Let's use a simple, fast, valid initialization:
    # A straight line is valid. Energy is 0.
    # But SA from a straight line is slow.
    # Let's use a "random walk with backtracking" properly implemented.
    
    def dfs_walk(idx, x, y, z, path, occ):
        if idx == n:
            return True
        path.append((x, y, z))
        occ[(x, y, z)] = True
        neighbors = list(_neighbours((x, y, z)))
        rng.shuffle(neighbors)
        for nx, ny, nz in neighbors:
            if (nx, ny, nz) not in occ:
                if dfs_walk(idx + 1, nx, ny, nz, path, occ):
                    return True
        path.pop()
        del occ[(x, y, z)]
        return False

    path = []
    occ = {}
    if dfs_walk(0, 0, 0, 0, path, occ):
        return path
    else:
        # Should not happen for small n, but fallback
        return [(i, 0, 0) for i in range(n)]

def _optimize_chain(sequence, chain, dimension, rng, time_limit):
    """
    Optimized Simulated Annealing with Pivot Moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    # Use a list for chain coords for fast access
    # Use a set for occupied for fast lookup
    
    # We need to be careful with the occupied set when moving segments.
    
    current_chain = chain[:]
    current_energy = _calc_energy(sequence, current_chain)
    best_energy = current_energy
    best_chain = current_chain[:]
    
    start_time = time.time()
    T = 1.0
    # Simple cooling: T = T0 * exp(-t/tau)
    tau = time_limit / 10.0
    
    while time.time() - start_time < time_limit:
        # Generate a random pivot move
        # Pick a bond index k (1 to n-1)
        # Rotate the segment chain[k:] around the bond (k-1, k)
        # Or pick an internal segment. Suffix moves are simpler and effective.
        
        if n < 4:
            k = 1
        else:
            # Weight towards shorter segments? Or uniform?
            # Uniform k in [1, n-1]
            k = rng.randint(1, n - 1)
            
        # Pivot is at k-1, axis is bond k-1 to k
        pivot_idx = k - 1
        pivot = current_chain[pivot_idx]
        
        # Determine axis
        p1 = current_chain[pivot_idx]
        p2 = current_chain[pivot_idx + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        
        if dx != 0:
            axis = 0
        elif dy != 0:
            axis = 1
        else:
            axis = 2
            
        # Choose rotation: 90, 180, 270
        # We need rotation matrices for the specific axis.
        # Precompute or calculate on the fly?
        # Let's define the 3 possible new positions for the segment start (k) relative to pivot.
        # Actually, it's easier to apply a rotation matrix to the whole segment.
        
        # Let's hardcode the 3 rotation results for the segment relative to the pivot.
        # If axis is X (dx!=0):
        #   The segment moves in the YZ plane.
        #   The bond vector is (1,0,0) or (-1,0,0).
        #   Rotating 90 deg around X: (y,z) -> (-z, y) or (z, -y)?
        #   Standard R_x(90): y' = -z, z' = y.
        #   R_x(180): y' = -y, z' = -z.
        #   R_x(270): y' = z, z' = -y.
        
        # We will generate the new segment coordinates.
        
        if axis == 0:
            # Rotate around X
            # y' = -z, z' = y (90)
            # y' = -y, z' = -z (180)
            # y' = z, z' = -y (270)
            rot_type = rng.randint(0, 2)
        elif axis == 1:
            # Rotate around Y
            # x' = z, z' = -x (90)
            # x' = -x, z' = -z (180)
            # x' = -z, z' = x (270)
            rot_type = rng.randint(0, 2)
        else:
            # Rotate around Z
            # x' = -y, y' = x (90)
            # x' = -x, y' = -y (180)
            # x' = y, y' = -x (270)
            rot_type = rng.randint(0, 2)
            
        segment = current_chain[k:]
        
        # Remove segment from occupied set
        # We need a fast way to check collisions.
        # Let's build a set of occupied points for the fixed part (0 to k-1).
        # And check the new segment against it.
        
        fixed_occupied = set(current_chain[:k])
        
        new_segment = []
        valid = True
        
        for p in segment:
            # Relative to pivot
            rx = p[0] - pivot[0]
            ry = p[1] - pivot[1]
            rz = p[2] - pivot[2]
            
            if axis == 0:
                if rot_type == 0: # 90
                    nry = -rz
                    nrz = ry
                elif rot_type == 1: # 180
                    nry = -ry
                    nrz = -rz
                else: # 270
                    nry = rz
                    nrz = -ry
                new_segment.append((pivot[0] + rx, pivot[1] + nry, pivot[2] + nrz))
                
            elif axis == 1:
                if rot_type == 0: # 90
                    nrx = rz
                    nrz = -rx
                elif rot_type == 1: # 180
                    nrx = -rx
                    nrz = -rz
                else: # 270
                    nrx = -rz
                    nrz = rx
                new_segment.append((pivot[0] + nrx, pivot[1] + ry, pivot[2] + nrz))
                
            else: # axis 2
                if rot_type == 0: # 90
                    nrx = -ry
                    nry = rx
                elif rot_type == 1: # 180
                    nrx = -rx
                    nry = -ry
                else: # 270
                    nrx = ry
                    nry = -rx
                new_segment.append((pivot[0] + nrx, pivot[1] + nry, pivot[2] + rz))
                
            # Check collision with fixed part
            if new_segment[-1] in fixed_occupied:
                valid = False
                break
            # Check collision with other parts of the new segment (self-collision within segment)
            # Since the segment is rigid, relative distances are preserved.
            # The only possible self-collisions are with the fixed part.
            # Wait, if the segment folds back on itself? 
            # A rotation around a bond preserves the self-avoiding property of the segment itself?
            # Yes, if the segment was self-avoiding, it remains self-avoiding after rotation.
            # So we only need to check against the fixed part.
            
        if not valid:
            continue
            
        # Calculate Delta Energy
        # Contacts involving the segment and the fixed part change.
        # Contacts within the segment remain the same.
        # Contacts within the fixed part remain the same.
        
        # Old contacts:
        old_contacts = 0
        for i, p in enumerate(segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in fixed_occupied:
                        nb_idx = current_chain.index(nb) # O(N) lookup, slow!
                        # We need a map from pos to index for the fixed part.
                        pass
        
        # This index lookup is O(N). We need a dict.
        # Let's build the dict once per move? That's O(N) per move. 
        # Total moves ~ 10000. N ~ 50. 500k ops. Acceptable.
        
        # Let's optimize: 
        # We already have `fixed_occupied` as a set. 
        # We need to know the index to check `is_h`.
        # We can build `fixed_map = {pos: idx for idx, pos in enumerate(current_chain[:k])}`
        
        fixed_map = {pos: idx for idx, pos in enumerate(current_chain[:k])}
        
        old_contacts = 0
        for i, p in enumerate(segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in fixed_map:
                        nb_idx = fixed_map[nb]
                        if is_h[nb_idx]:
                            old_contacts += 1
                            
        new_contacts = 0
        for i, p in enumerate(new_segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in fixed_map:
                        nb_idx = fixed_map[nb]
                        if is_h[nb_idx]:
                            new_contacts += 1
                            
        delta_e = -(new_contacts - old_contacts)
        
        # Acceptance
        elapsed = time.time() - start_time
        T = 1.0 * math.exp(-elapsed / tau)
        
        accept = False
        if delta_e < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta_e / T)
                accept = rng.random() < prob
                
        if accept:
            # Update chain
            current_chain[k:] = new_segment
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    # Global deadline management
    # 240s budget for all. 26 sequences. ~9s each.
    # Use 8s to be safe.
    PER_CALL_TIME_LIMIT = 8.0
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    
    rng = random.Random(12345 + len(sequence))
    
    # 1. Initial Chain
    initial_chain = _generate_initial_chain(sequence, dimension, rng)
    
    # 2. Optimization
    # Allocate remaining time to SA
    time_elapsed = time.time() - start_time
    sa_time = deadline - time_elapsed - 0.5 # 0.5s margin
    
    if sa_time > 0.1:
        best_chain = _optimize_chain(sequence, initial_chain, dimension, rng, sa_time)
    else:
        best_chain = initial_chain
        
    # Ensure return type is list of lists of ints
    return [list(p) for p in best_chain]
