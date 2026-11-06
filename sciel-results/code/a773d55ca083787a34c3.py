"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _generate_initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk using a greedy random walk."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            # Dead end, try to backtrack or just fail this attempt
            # For simplicity in initialization, we return None if stuck
            # But a simple random walk rarely gets stuck for N < 100 in 3D
            # unless it coils tightly. 
            # If stuck, we can try a different random path or just return what we have?
            # The task requires a full chain. If we can't complete it, we should try again.
            return None
        
        # Randomly pick a site
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _calculate_energy(sequence, chain):
    """Calculate the energy of a given chain conformation."""
    if not chain:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if they are not adjacent in the sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            # To avoid double counting, only count if i < j
                            if i < j:
                                contacts += 1
    return -contacts

def _try_move_kink(chain, sequence, occupied, idx, rng):
    """
    Try to move the residue at index 'idx' to a neighboring empty site.
    Returns (new_chain, new_occupied, new_energy_delta) if successful, else None.
    """
    if idx <= 0 or idx >= len(chain) - 1:
        return None
        
    p = chain[idx]
    neighbors = list(_neighbours(p))
    rng.shuffle(neighbors)
    
    for new_p in neighbors:
        # Check if new position is empty
        if new_p in occupied:
            continue
        
        # Check if new position collides with immediate neighbors in the chain
        # Neighbors in chain are idx-1 and idx+1
        prev_p = chain[idx - 1]
        next_p = chain[idx + 1]
        
        # The new position must not be the same as prev or next (obvious since they are occupied)
        # But we must ensure the bond lengths are still 1? 
        # Moving to a neighbor of p ensures dist(p, new_p) = 1.
        # However, the new position must be connected to prev and next?
        # No, a "kink" move usually implies moving the whole tail? 
        # No, a simple "pivot" or "local move" moves a single residue.
        # If we move residue i, it must be adjacent to i-1 and i+1.
        # So new_p must be a neighbor of prev_p AND a neighbor of next_p.
        
        if new_p not in _neighbours(prev_p):
            continue
        if new_p not in _neighbours(next_p):
            continue
            
        # If we pass these checks, the move is structurally valid.
        # We don't need to recalculate full energy here, just return the new state.
        # We need to update occupied set.
        
        new_occupied = set(occupied)
        new_occupied.remove(p)
        new_occupied.add(new_p)
        
        new_chain = chain.copy()
        new_chain[idx] = new_p
        
        return new_chain, new_occupied

def _try_move_pivot(chain, sequence, occupied, idx, rng):
    """
    Try a pivot move at index 'idx'.
    Rotate the tail (idx+1 to end) around the bond (idx, idx+1).
    """
    if idx <= 0 or idx >= len(chain) - 1:
        return None
        
    p0 = chain[idx]
    p1 = chain[idx + 1]
    
    # Vector from p0 to p1
    v = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    
    # Find two orthogonal vectors to v to define the rotation plane
    # v is a unit vector along one axis.
    # We need to rotate the tail.
    # The tail starts at p1.
    # We can rotate the entire tail by 90, 180, 270 degrees around the axis defined by v.
    
    # Determine the axis of v
    axis = None
    for i in range(3):
        if abs(v[i]) == 1:
            axis = i
            break
    if axis is None:
        return None
        
    # The other two axes are the ones we can rotate in
    other_axes = [i for i in range(3) if i != axis]
    ax1, ax2 = other_axes
    
    # Generate the 4 possible rotated tails
    # Original tail
    original_tail = chain[idx+1:]
    
    # We need to rotate each point in the tail around the line passing through p0 and p1.
    # Since the lattice is cubic, we can just try the 4 possible directions for the first bond of the tail?
    # No, the rest of the tail must rotate consistently.
    
    # A simpler way: 
    # The bond p0->p1 is fixed.
    # The next bond p1->p2 can go in 4 directions relative to p0->p1 (including backtracking which is invalid).
    # Actually, in a cubic lattice, if p0->p1 is along X, p1->p2 can be +X, -X, +Y, -Y, +Z, -Z.
    # -X is backtracking (invalid).
    # So 5 options.
    # But a pivot move rotates the WHOLE tail.
    
    # Let's implement a rotation of the tail coordinates.
    # Center of rotation is p1? No, the bond p0-p1 is the hinge.
    # We rotate the vector from p1 to each subsequent point.
    
    # Let's just try the 4 distinct rotations of the tail.
    # We can generate the rotated tail by rotating the vector components.
    
    # Define rotation matrices for 90 deg around X, Y, Z.
    # But we need rotation around the specific axis 'axis'.
    
    # If axis == 0 (X-axis):
    # Rotate Y,Z
    # (y, z) -> (-z, y) for 90 deg
    # (y, z) -> (z, -y) for 270 deg
    # (y, z) -> (-y, -z) for 180 deg
    
    # We will generate 3 new tails (90, 180, 270).
    
    new_tails = []
    
    for angle in [90, 180, 270]:
        new_tail = []
        for p in chain[idx+1:]:
            # Relative to p1
            rel = (p[0] - p1[0], p[1] - p1[1], p[2] - p1[2])
            
            # Rotate rel around axis
            new_rel = list(rel)
            
            if axis == 0: # X-axis
                if angle == 90:
                    new_rel[1] = -rel[2]
                    new_rel[2] = rel[1]
                elif angle == 180:
                    new_rel[1] = -rel[1]
                    new_rel[2] = -rel[2]
                elif angle == 270:
                    new_rel[1] = rel[2]
                    new_rel[2] = -rel[1]
            elif axis == 1: # Y-axis
                if angle == 90:
                    new_rel[0] = rel[2]
                    new_rel[2] = -rel[0]
                elif angle == 180:
                    new_rel[0] = -rel[0]
                    new_rel[2] = -rel[2]
                elif angle == 270:
                    new_rel[0] = -rel[2]
                    new_rel[2] = rel[0]
            elif axis == 2: # Z-axis
                if angle == 90:
                    new_rel[0] = -rel[1]
                    new_rel[1] = rel[0]
                elif angle == 180:
                    new_rel[0] = -rel[0]
                    new_rel[1] = -rel[1]
                elif angle == 270:
                    new_rel[0] = rel[1]
                    new_rel[1] = -rel[0]
            
            # Check if rotation resulted in integer coordinates (it should, since we swap and negate integers)
            # And check if the new position is valid (no collision)
            # We must check collisions for the ENTIRE new tail against the occupied set (excluding the tail itself)
            # AND against the new tail itself.
            
            new_p = (p1[0] + new_rel[0], p1[1] + new_rel[1], p1[2] + new_rel[2])
            new_tail.append(new_p)
            
        # Now check validity of this new tail
        valid = True
        occupied_set = set(occupied)
        # Remove the old tail from occupied to check collisions
        for p in chain[idx+1:]:
            occupied_set.remove(p)
        
        # Add new tail to a temp set to check self-collision
        new_tail_set = set(new_tail)
        if len(new_tail_set) != len(new_tail):
            valid = False
        
        if valid:
            # Check collision with rest of chain
            for p in new_tail:
                if p in occupied_set:
                    valid = False
                    break
        
        if valid:
            new_chain = chain[:idx+1] + new_tail
            new_occupied = occupied_set | new_tail_set
            return new_chain, new_occupied
            
    return None

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Kink and Pivot moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds per sequence
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start SA
    num_starts = 50
    
    for s in range(num_starts):
        if time.time() > deadline - 1.0:
            break
            
        # Generate initial chain
        chain = _generate_initial_chain(sequence, dimension, rng)
        if chain is None:
            continue
            
        occupied = {p: i for i, p in enumerate(chain)}
        energy = _calculate_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain.copy()
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        # Number of steps per start
        # Adjust based on time
        steps = 2000
        
        for step in range(steps):
            if time.time() > deadline - 0.5:
                break
                
            move_type = rng.random()
            if move_type < 0.5:
                # Kink move
                idx = rng.randint(1, n - 2)
                result = _try_move_kink(chain, sequence, occupied, idx, rng)
                if result:
                    new_chain, new_occupied = result
                    new_energy = _calculate_energy(sequence, new_chain)
                    delta_e = new_energy - energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        chain = new_chain
                        occupied = new_occupied
                        energy = new_energy
                        if energy < best_energy:
                            best_energy = energy
                            best_chain = chain.copy()
            else:
                # Pivot move
                idx = rng.randint(1, n - 2)
                result = _try_move_pivot(chain, sequence, occupied, idx, rng)
                if result:
                    new_chain, new_occupied = result
                    new_energy = _calculate_energy(sequence, new_chain)
                    delta_e = new_energy - energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        chain = new_chain
                        occupied = new_occupied
                        energy = new_energy
                        if energy < best_energy:
                            best_energy = energy
                            best_chain = chain.copy()
                            
            T *= cooling
            if T < T_min:
                T = T_min
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
