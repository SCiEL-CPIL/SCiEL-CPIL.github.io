"""HP lattice protein folding using Simulated Annealing with Pivot Rotations."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.90  # Use 90% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer positions with more H-H contacts and more free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _pivot_move(chain, pos_map, pivot_idx, rng):
    """
    Perform a pivot move: rotate a subchain around a pivot point.
    Returns (new_chain, new_pos_map, delta_energy) or None if invalid.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pivot must be at least 1 away from ends to have a subchain to rotate
    if pivot_idx < 1 or pivot_idx > n - 2:
        return None
    
    # The subchain to rotate is chain[pivot_idx+1:]
    # We rotate this subchain around chain[pivot_idx]
    # The vector from pivot to next point is v = chain[pivot_idx+1] - chain[pivot_idx]
    # We generate a new vector v' by rotating v.
    # To keep it simple and valid on lattice, we pick a new direction for the first step
    # and then extend the rest of the subchain relative to that? No, that's not a rigid rotation.
    
    # True pivot move:
    # Let p0 = chain[pivot_idx]
    # Let sub = chain[pivot_idx+1:]
    # The subchain is a set of vectors relative to p0: v_k = sub[k] - p0
    # We apply a rotation matrix R to all v_k to get v'_k.
    # The new positions are p0 + v'_k.
    # We need R such that v'_0 is a unit vector and the whole chain remains self-avoiding.
    
    # Simple approach: Try all 6 possible directions for the new first step of the subchain.
    # Then, for the rest of the subchain, we need to apply a rotation that maps the old first step to the new first step.
    # This is complex. A simpler "quantum-inspired" or "tunneling" move might be better:
    # Randomly pick a segment and try to flip it or rotate it 180 degrees?
    
    # Let's implement a simpler move: "Segment Flip" or "Crankshaft".
    # Or, let's implement a proper 3D pivot rotation.
    
    # Given the complexity of rigid rotation in code without numpy (though numpy is available),
    # let's use a simpler heuristic move that is known to work well:
    # "Slide" move: Move a single residue to an adjacent empty site.
    # "Pivot" move: Rotate a subchain.
    
    # For this implementation, I will use a combination of:
    # 1. Single residue move (translation to neighbor)
    # 2. Pivot move (rotate subchain by 180 degrees around the axis defined by pivot and next point? No.)
    
    # Let's use a standard Pivot Move implementation:
    # Pick pivot i.
    # Let v = chain[i+1] - chain[i].
    # Choose a new direction v' from the 6 neighbors of 0 (unit vectors).
    # If v' == v, no change.
    # If v' == -v, 180 degree flip.
    # Otherwise, 90 degree rotation.
    # We need to determine the rotation matrix that maps v to v'.
    # There are multiple such matrices. We can pick one.
    # Then apply it to all subsequent points in the subchain.
    
    # This is computationally expensive to do correctly for every move.
    # Given the time budget, I will implement a simpler "Quantum Tunneling" approximation:
    # Randomly select a segment [i, j] and try to "flip" it or "rotate" it 180 degrees.
    
    # Actually, the most effective simple move is the "Pivot" where we rotate the tail.
    # Let's try a simpler "Random Walk" perturbation:
    # Pick a random index i.
    # Try to move chain[i] to a random neighbor site.
    # This is the "single residue move".
    
    # To satisfy the "tunneling" requirement, I will add a "Segment Rotation" move.
    # Pick two indices i < j.
    # Rotate the segment chain[i:j] around chain[i-1] (if i>0) or chain[0].
    
    # For simplicity and robustness, I will implement:
    # 1. Single Residue Move (Slide)
    # 2. Pivot Move (Rotate tail from i)
    
    # Pivot Move Implementation:
    # To rotate the tail from i, we need to map the vector v = chain[i] - chain[i-1] to a new vector v'.
    # We choose v' randomly from the 5 other unit vectors.
    # We construct a rotation matrix R that maps v to v'.
    # We apply R to all points chain[i:] relative to chain[i-1].
    
    # Helper to get rotation matrix mapping u to v (unit vectors)
    def _rotation_matrix(u, v):
        # u, v are tuples (x, y, z)
        u = [float(x) for x in u]
        v = [float(x) for x in v]
        
        # If u == v, return identity
        if u[0] == v[0] and u[1] == v[1] and u[2] == v[2]:
            return [[1,0,0],[0,1,0],[0,0,1]]
        
        # If u == -v, 180 degree rotation around any orthogonal axis
        # Let's pick axis perpendicular to u
        # Cross product of u and some arbitrary vector
        # Simpler: If u == -v, we can just negate the coordinates? No, that's a reflection.
        # A 180 degree rotation around the axis perpendicular to u.
        # Let axis = u x (1,0,0) if u != (1,0,0) else u x (0,1,0)
        
        # General formula for rotation matrix mapping u to v:
        # R = I + [w]x + [w]x^2 * (1/(1+c))
        # where w = u x v, c = u . v
        
        def cross(a, b):
            return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
        
        def dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        
        c = dot(u, v)
        if c < -0.9999:
            # u and v are opposite
            # Find an axis perpendicular to u
            if abs(u[0]) < 0.9:
                axis = cross(u, [1, 0, 0])
            else:
                axis = cross(u, [0, 1, 0])
            # Normalize axis
            norm = math.sqrt(sum(x*x for x in axis))
            if norm < 1e-6:
                return [[1,0,0],[0,1,0],[0,0,1]]
            axis = [x/norm for x in axis]
            # 180 degree rotation around axis
            # R = 2 * axis * axis^T - I
            ax, ay, az = axis
            return [
                [2*ax*ax - 1, 2*ax*ay, 2*ax*az],
                [2*ay*ax, 2*ay*ay - 1, 2*ay*az],
                [2*az*ax, 2*az*ay, 2*az*az - 1]
            ]
        
        w = cross(u, v)
        wx, wy, wz = w
        wx_x = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
        
        # R = I + wx_x + wx_x @ wx_x * (1 / (1 + c))
        # wx_x @ wx_x = w w^T - (w . w) I = w w^T - (1 - c^2) I
        # So term is (w w^T - (1-c^2)I) * (1/(1+c))
        # Let's just compute matrix multiplication
        
        def mat_mul(A, B):
            return [
                [sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]
        
        wx_x_sq = mat_mul(wx_x, wx_x)
        
        R = [[0.0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                R[i][j] = (1 if i==j else 0) + wx_x[i][j] + wx_x_sq[i][j] * (1.0 / (1.0 + c))
        
        return R
    
    def apply_rotation(R, vec):
        return (
            R[0][0]*vec[0] + R[0][1]*vec[1] + R[0][2]*vec[2],
            R[1][0]*vec[0] + R[1][1]*vec[1] + R[1][2]*vec[2],
            R[2][0]*vec[0] + R[2][1]*vec[1] + R[2][2]*vec[2]
        )
    
    # Pivot index
    i = pivot_idx
    if i == 0:
        # Rotate whole chain? No, fix first point.
        return None
    
    p_prev = chain[i-1]
    p_curr = chain[i]
    
    v_old = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1], p_curr[2]-p_prev[2])
    
    # Choose new direction
    new_dirs = list(_neighbours((0,0,0)))
    # Remove v_old
    new_dirs = [d for d in new_dirs if d != v_old]
    if not new_dirs:
        return None
    
    v_new = new_dirs[rng.randint(0, len(new_dirs)-1)]
    
    # Get rotation matrix
    R = _rotation_matrix(v_old, v_new)
    
    # Apply to subchain
    new_chain = list(chain[:i])
    new_pos_map = dict(pos_map)
    # Remove old positions of subchain
    for k in range(i, n):
        del new_pos_map[chain[k]]
    
    # Calculate new positions
    valid = True
    new_sub_points = []
    for k in range(i, n):
        # Vector from p_prev to chain[k]
        vec = (chain[k][0]-p_prev[0], chain[k][1]-p_prev[1], chain[k][2]-p_prev[2])
        new_vec = apply_rotation(R, vec)
        # Round to nearest integer (should be exact if R is orthogonal and vec is integer)
        new_p = (p_prev[0] + round(new_vec[0]), p_prev[1] + round(new_vec[1]), p_prev[2] + round(new_vec[2]))
        
        # Check self-avoidance
        if new_p in new_pos_map:
            valid = False
            break
        
        # Check if it's an integer lattice point (sanity check)
        if new_p[0] != new_vec[0] + p_prev[0] or new_p[1] != new_vec[1] + p_prev[1] or new_p[2] != new_vec[2] + p_prev[2]:
             # If floating point issues, try rounding
             pass
             
        new_sub_points.append(new_p)
        new_pos_map[new_p] = k
        
    if not valid:
        return None
    
    new_chain.extend(new_sub_points)
    
    return new_chain, new_pos_map

def _simulated_annealing(sequence, chain, dimension, rng, time_budget):
    """Simulated Annealing with Pivot Moves."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    current_chain = list(chain)
    current_pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Choose move: 80% Pivot, 20% Single Residue
        if rng.random() < 0.8:
            # Pivot Move
            # Pick pivot index between 1 and n-2
            i = rng.randint(1, n-2)
            result = _pivot_move(current_chain, current_pos_map, i, rng)
            if result:
                new_chain, new_pos_map = result
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_pos_map = new_pos_map
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        else:
            # Single Residue Move
            i = rng.randint(1, n-2)
            p_i = current_chain[i]
            
            for site in _neighbours(p_i):
                if site in current_pos_map:
                    continue
                if i > 0:
                    prev = current_chain[i-1]
                    if abs(site[0]-prev[0]) + abs(site[1]-prev[1]) + abs(site[2]-prev[2]) != 1:
                        continue
                if i < n-1:
                    next_p = current_chain[i+1]
                    if abs(site[0]-next_p[0]) + abs(site[1]-next_p[1]) + abs(site[2]-next_p[2]) != 1:
                        continue
                
                # Valid move
                new_chain = list(current_chain)
                new_chain[i] = site
                new_pos_map = dict(current_pos_map)
                del new_pos_map[p_i]
                new_pos_map[site] = i
                
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_pos_map = new_pos_map
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_chain

def fold(sequence, dimension):
    """Fold a protein sequence on the simple cubic lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Allocate time based on sequence length
    # Longer sequences need more time
    time_per_seq = 10 + n * 0.1  # Rough heuristic
    
    num_trials = 5
    
    for trial in range(num_trials):
        if _is_deadline_exceeded():
            break
        
        # Generate initial chain
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Simulated Annealing improvement
        # Allocate remaining time / remaining trials
        remaining_time = _get_deadline() - time.time()
        trials_left = num_trials - trial
        sa_time = max(1.0, remaining_time / trials_left * 0.8)
        
        improved = _simulated_annealing(sequence, chain, dimension, rng, sa_time)
        improved_energy = _energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # If no valid chain found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
