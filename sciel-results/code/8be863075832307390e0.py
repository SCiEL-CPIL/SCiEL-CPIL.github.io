import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
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

def _rotate_90(vec, axis):
    # vec is (vx, vy, vz), axis is (ax, ay, az) which is a unit vector (dx, dy, dz)
    # We perform a 90 degree rotation around the axis.
    # There are two directions: +90 and -90. We need to try both or pick one?
    # In pivot moves, we usually pick one of the 3 possible new orientations for the bond.
    # Actually, for a pivot at i, we rotate the segment i..N-1 around the bond (i-1, i).
    # The bond vector is fixed. The rest rotates.
    # There are 3 possible rotations of the tail that keep the lattice structure?
    # No, the bond (i-1, i) is fixed. The next bond (i, i+1) can go to 3 other neighbors of i?
    # No, (i, i+1) is part of the rotating segment.
    # The rotation is around the axis defined by (chain[i] - chain[i-1]).
    # Let's implement the rotation of a vector V around a unit axis A by 90 degrees.
    # Formula: V' = V cos(90) + (A x V) sin(90) + A(A.V)(1-cos(90))
    # V' = A x V  (for +90) or V x A (for -90)?
    # Let's just calculate the cross product.
    vx, vy, vz = vec
    ax, ay, az = axis
    
    # Cross product A x V
    cx = ay * vz - az * vy
    cy = az * vx - ax * vz
    cz = ax * vy - ay * vx
    
    return (cx, cy, cz)

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0: return []
    
    rng = random.Random(12345 + N)
    
    # 1. Initialize
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
        
    occupied = set(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_e = _energy(sequence, chain)
    best_e = current_e
    best_chain = list(chain)
    
    start_time = time.time()
    # Budget is 240s for all sequences. Assume ~9s per sequence.
    # Leave safety margin.
    deadline = start_time + 8.5
    
    # Simulated Annealing Parameters
    # Based on knowledge: T0 = -best_energy * 0.5
    # If best_energy is 0, T0 = 0.5? Let's use a fixed reasonable T0 based on chain length or max possible contacts.
    # Max contacts is roughly N/2. Let's say T0 = 5.0 initially, cool down.
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999 # Adjust based on steps
    
    # Precompute H indices for faster checks?
    # For now, direct access is fine.
    
    while time.time() < deadline:
        # Choose pivot index i (1 to N-2)
        i = rng.randint(1, N - 2)
        
        p_prev = chain[i-1]
        p_curr = chain[i]
        
        # Axis is the bond vector from prev to curr
        axis = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
        
        # We rotate the segment chain[i:] around the axis.
        # We need to try the 3 possible 90-degree rotations?
        # Actually, rotating by 90 degrees around the bond axis changes the direction of the next bond.
        # There are 3 possible new directions for the bond (i, i+1) that are not the old one?
        # No, the bond (i, i+1) is rotated.
        # Let's just pick one of the two 90-degree rotations (+90 or -90).
        # Actually, for a cubic lattice, rotating a vector by 90 degrees around a unit axis
        # results in another integer vector.
        
        # Let's generate the new segment
        new_seg = []
        valid = True
        
        # We need to rotate each vector (chain[j] - chain[i-1])?
        # No, it's easier to rotate relative to the pivot point chain[i-1]?
        # The pivot is the bond (i-1, i). The point chain[i-1] is fixed. The point chain[i] is fixed.
        # Wait, if chain[i-1] and chain[i] are fixed, the axis is fixed.
        # The rest of the chain rotates.
        # Let's rotate the vectors from chain[i-1] to chain[j] for j >= i.
        # But chain[i] is on the axis, so it stays.
        
        # Let's store the old positions of the moving part to check collisions
        moving_old = chain[i:]
        
        # Calculate new positions
        # For each point P in moving_old:
        # V = P - p_prev
        # V_rot = rotate_90(V, axis)
        # P_new = p_prev + V_rot
        
        # We need to try both +90 and -90? Or just one?
        # Usually, we pick a random rotation direction.
        sign = 1 if rng.random() < 0.5 else -1
        
        # Pre-calculate cross products? No, just do it.
        
        # Check for collisions with the fixed part (0..i-1) and within the new segment
        # The new segment must not collide with the fixed part.
        # The new segment must not self-intersect (which is guaranteed if the rotation is rigid and the old segment was SAW?
        # No, rigid rotation preserves distances, so internal collisions are preserved.
        # But it might collide with the fixed part.
        
        # Also, we must ensure the new segment doesn't overlap with itself?
        # Rigid rotation of a SAW is a SAW. So internal collisions are not an issue.
        # We only need to check collisions with the fixed part (indices 0 to i-1).
        
        # However, we also need to make sure the new segment doesn't collide with the *old* positions of the moving part?
        # No, because the old positions are vacated.
        
        # So, check if any P_new is in occupied_fixed = {chain[0]...chain[i-1]}
        
        fixed_set = set(chain[:i])
        
        # Optimization: Instead of checking all, we can just build the new set and check intersection.
        # But we need to update 'occupied' and 'where' if we accept.
        
        # Let's generate the new segment positions
        new_positions = []
        for j in range(i, N):
            p_old = chain[j]
            v = (p_old[0] - p_prev[0], p_old[1] - p_prev[1], p_old[2] - p_prev[2])
            # Rotate v around axis by 90 degrees * sign
            # My _rotate_90 does A x V.
            # If sign is -1, we do V x A which is -(A x V).
            rx, ry, rz = _rotate_90(v, axis)
            if sign == -1:
                rx, ry, rz = -rx, -ry, -rz
            
            p_new = (p_prev[0] + rx, p_prev[1] + ry, p_prev[2] + rz)
            
            # Check collision with fixed part
            if p_new in fixed_set:
                valid = False
                break
            
            new_positions.append(p_new)
        
        if not valid:
            continue
            
        # Calculate Delta Energy
        # Only contacts involving the boundary (i-1, i) and contacts between fixed and moving parts change?
        # Internal contacts within fixed part: Unchanged.
        # Internal contacts within moving part: Unchanged (rigid rotation).
        # Cross contacts (u < i, v >= i): Change.
        
        # We need to count H-H contacts between chain[0..i-1] and chain[i..N-1].
        # Old cross contacts:
        old_cross = 0
        for u in range(i):
            if sequence[u] != 'H': continue
            pu = chain[u]
            for v in range(i, N):
                if sequence[v] != 'H': continue
                pv = chain[v]
                if abs(pu[0]-pv[0]) + abs(pu[1]-pv[1]) + abs(pu[2]-pv[2]) == 1:
                    old_cross += 1
                    
        # New cross contacts:
        new_cross = 0
        for u in range(i):
            if sequence[u] != 'H': continue
            pu = chain[u]
            for k, pv_new in enumerate(new_positions):
                v_idx = i + k
                if sequence[v_idx] != 'H': continue
                if abs(pu[0]-pv_new[0]) + abs(pu[1]-pv_new[1]) + abs(pu[2]-pv_new[2]) == 1:
                    new_cross += 1
                    
        delta_e = -(new_cross - old_cross)
        
        # Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Update chain
            # Remove old moving part from occupied
            for p in moving_old:
                occupied.remove(p)
                
            # Add new moving part
            for p in new_positions:
                occupied.add(p)
                
            # Update chain
            chain[i:] = new_positions
            
            # Update where (only for moving part)
            for k, p in enumerate(new_positions):
                where[p] = i + k
                
            current_e += delta_e
            
            if current_e < best_e:
                best_e = current_e
                best_chain = list(chain)
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
    return best_chain
