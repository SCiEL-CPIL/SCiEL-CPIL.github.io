"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-restart with Beam Search initialization and Simulated Annealing local optimization.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None
_safety_margin = 15.0


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - _safety_margin


def _neighbours(point):
    """Yield 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _calculate_energy(sequence, chain):
    """Calculate the total energy of a chain. Energy = - (number of H-H contacts)."""
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


def _beam_search_init(sequence, dimension, width=50, rng=None):
    """
    Generate initial chains using beam search.
    Returns a list of candidate chains.
    """
    if rng is None:
        rng = random.Random()
    
    start = (0,) * dimension
    # State: (score, chain, occupied_dict)
    # Score: -current_energy + heuristic
    # We want to minimize energy, so maximize score.
    # Heuristic: count potential H-H contacts.
    
    current_beam = []
    start_occupied = {start: 0}
    current_beam.append((0, [start], start_occupied))
    
    n = len(sequence)
    for i in range(1, n):
        next_beam = []
        for score, chain, occupied in current_beam:
            last_pos = chain[-1]
            candidates = []
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                # Calculate gain in energy (contacts)
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                gain += 1
                
                # Heuristic: number of free neighbours (to avoid dead ends)
                free_space = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_space += 1
                
                # Score for beam search:
                # We want high gain (more contacts) and high free space (less trapped).
                # Energy is negative, so gain is positive reduction in energy.
                # Let's use: score = current_score + gain * 2 + free_space * 0.1
                # This prioritizes contacts but avoids dead ends.
                new_score = score + gain * 2.0 + free_space * 0.1
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                candidates.append((new_score, new_chain, new_occupied))
        
        if not candidates:
            return []
        
        # Keep top 'width' candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        current_beam = candidates[:width]
        
        if _time_remaining() < 5.0:
             # If time is running out during init, return what we have or a simple line
             # But beam search is fast, so this is unlikely unless width is huge
             break

    # Return the chains from the final beam
    return [chain for _, chain, _ in current_beam]


def _simulated_annealing(sequence, chain, rng, max_steps=2000, t0=1.0, cooling=0.99):
    """
    Perform Simulated Annealing on a given chain.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 5:
        return chain
    
    # Pre-compute residue types for faster access
    res = sequence
    
    # Current state
    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _calculate_energy(res, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t = t0
    step = 0
    
    # Helper to check if a move is valid (self-avoiding)
    # For a move involving a segment, we need to check if the new positions collide
    # with the rest of the chain.
    
    def _is_valid_move(moved_indices, new_positions):
        """
        moved_indices: list of indices that are moving
        new_positions: list of new tuples corresponding to moved_indices
        """
        # Create a set of new positions for O(1) lookup
        new_pos_set = set(new_positions)
        
        # Check if any new position is occupied by an index NOT in moved_indices
        for i in moved_indices:
            if i in current_occupied:
                old_pos = current_occupied[i]
                # If old_pos is not in new_pos_set, it becomes free.
                # If new_pos is in current_occupied and the index at current_occupied[new_pos] is not in moved_indices, it's a collision.
        
        # More efficient check:
        # 1. Build set of indices being moved
        moving_set = set(moved_indices)
        
        # 2. For each new position, check if it is occupied by a non-moving index
        for new_p in new_positions:
            if new_p in current_occupied:
                occ_idx = current_occupied[new_p]
                if occ_idx not in moving_set:
                    return False
        
        # 3. Also need to ensure that the new positions are unique among themselves
        # (handled by set construction if we check len)
        if len(new_pos_set) != len(new_positions):
            return False
            
        return True

    while step < max_steps:
        if _time_remaining() < 2.0:
            break
            
        # Choose a move type: Kink or Pivot
        move_type = rng.random()
        
        delta_e = 0
        valid = False
        new_chain = None
        new_occupied = None
        
        if move_type < 0.5:
            # Kink Move: Reverse segment [i+1, j]
            # Choose i, j such that 0 <= i < j < n
            # i is the pivot bond start, segment i+1..j moves
            if n < 4:
                continue
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            if j <= i + 1:
                continue
                
            # The segment to reverse is indices i+1 to j
            seg_indices = list(range(i + 1, j + 1))
            # New positions: chain[i+1..j] reversed
            # Original positions: current_chain[i+1], ..., current_chain[j]
            # New positions: current_chain[j], ..., current_chain[i+1]
            
            new_positions = [current_chain[k] for k in range(j, i, -1)]
            
            if _is_valid_move(seg_indices, new_positions):
                # Calculate Delta Energy
                # Energy changes only for contacts involving the moved segment
                # This is complex to calculate incrementally without a full map.
                # For simplicity and correctness, we might just recompute energy for small N
                # Or we can do a quick check.
                # Given the time constraints, let's try to be smarter.
                
                # Let's just update the chain and recompute energy if accepted?
                # No, SA needs delta E for acceptance probability.
                
                # Optimized Delta E calculation:
                # 1. Remove all contacts involving any index in seg_indices
                # 2. Add all contacts involving the new positions
                # This is still O(N) per move.
                
                # Let's use a simplified approach:
                # Since we are in a tight loop, let's try to calculate delta E efficiently.
                # However, implementing a fully incremental energy calculator is error-prone.
                # Let's assume N is up to 136. O(N) per move is acceptable if max_steps is reasonable.
                
                # Update chain
                new_chain = list(current_chain)
                for idx, pos in zip(seg_indices, new_positions):
                    new_chain[idx] = pos
                    
                # Update occupied dict
                new_occupied = dict(current_occupied)
                for idx in seg_indices:
                    # Remove old position mapping
                    old_pos = current_chain[idx]
                    # We can't just delete from dict easily if multiple keys map to same value? 
                    # No, keys are unique positions.
                    # We need to remove old_pos from the dict? No, the dict maps pos->index.
                    # So we need to update the values for the new positions.
                    # And remove the old positions from the dict?
                    # Actually, the dict `current_occupied` maps Position -> Index.
                    # When we move, the old positions become free, and new positions become occupied.
                    
                    # Let's rebuild the occupied dict for the moved parts?
                    # That's O(N).
                    
                    # Let's just recompute energy for the new chain.
                    # It's O(N*contacts) ~ O(N^2) in worst case, but usually sparse.
                    # For N=136, N^2 is ~18k. 2000 steps * 18k = 36M ops. Might be too slow in Python.
                    
                    # Let's try to optimize the energy calculation.
                    # We'll stick to a faster move: Pivot might be easier?
                    # Or we accept the cost of recomputation if we reduce max_steps.
                    
                    # Let's try a different tactic: 
                    # Use a set of occupied positions for quick collision check.
                    # And recompute energy only if the move is geometrically valid.
                    
                    # Let's refine _is_valid_move to also return the new occupied set?
                    # No, let's just recompute energy.
                    pass
                
                # To make this faster, let's limit max_steps based on N.
                # And use a faster energy check.
                
                new_energy = _calculate_energy(res, new_chain)
                delta_e = new_energy - current_energy
                valid = True
        else:
            # Pivot Move: Rotate segment [i+1, N-1] around bond i-i+1
            if n < 4:
                continue
            i = rng.randint(0, n - 2)
            if i == n - 2:
                continue # Rotating last residue around previous is just a kink of length 1? 
                         # Actually, rotating the last residue around the previous bond is a valid move.
                         # But if i is n-2, segment is just [n-1].
                         # Let's allow it.
            
            # Axis of rotation: vector from chain[i] to chain[i+1]
            p0 = current_chain[i]
            p1 = current_chain[i+1]
            
            # Determine the axis and the plane of rotation
            # In 3D, rotating around an axis defined by two points.
            # The axis is one of the 6 directions.
            # The perpendicular plane has 4 other directions.
            # We pick a random rotation: 90, 180, 270 degrees.
            
            # Vector v = p1 - p0
            v = [p1[k] - p0[k] for k in range(3)]
            
            # Find a vector perpendicular to v
            # If v is along x, perp can be y or z.
            # Let's pick a random perpendicular axis u.
            # Then rotate the segment by 90 deg around v?
            # No, the bond is fixed. The rest of the chain rotates around the bond axis.
            # The bond axis is v.
            # We need to rotate the vector from p1 to p2, etc., around v.
            
            # Simpler: 
            # The segment is p1, p2, ..., pn-1.
            # p1 is fixed relative to p0.
            # We rotate the entire tail (p1...pn-1) around the axis p0-p1.
            # Wait, p1 is part of the tail? 
            # Usually, pivot moves rotate the tail starting from i+1.
            # So p_{i+1} moves. The bond (i, i+1) rotates? No, the bond is the axis.
            # So p_{i+1} stays in the same line? No, p_{i+1} is on the axis.
            # So p_{i+1} doesn't move? 
            # If the axis is the bond (i, i+1), then p_{i+1} is on the axis.
            # So p_{i+1} stays fixed.
            # p_{i+2}, ..., p_{n-1} rotate around the axis.
            
            # Let's verify: 
            # Axis: p0 -> p1.
            # Points: p1, p2, ...
            # p1 is on the axis.
            # p2 is not.
            # Rotate p2 around axis p0-p1.
            
            # How to rotate a point around an axis in 3D integer lattice?
            # This is tricky. The result must be an integer lattice point.
            # Only 90, 180, 270 degree rotations around lattice axes preserve the lattice.
            # The axis v must be a lattice vector (dx, dy, dz) with one non-zero component?
            # No, the bond is always a unit step in one of the 6 directions.
            # So v is one of (1,0,0), (-1,0,0), etc.
            # So the axis is always aligned with one of the coordinate axes!
            
            # Great. So we can just swap coordinates.
            # If axis is X, rotate in YZ plane.
            # If axis is Y, rotate in XZ plane.
            # If axis is Z, rotate in XY plane.
            
            axis_coord = -1
            for k in range(3):
                if v[k] != 0:
                    axis_coord = k
                    break
            
            # Determine rotation angle: 90, 180, 270
            angle = rng.choice([1, 2, 3]) # 1=90, 2=180, 3=270
            
            # Indices to move: i+1 to n-1
            # But p_{i+1} is on the axis, so it doesn't move?
            # Wait, if we rotate around the bond (i, i+1), the bond itself is the axis.
            # So p_i and p_{i+1} are fixed.
            # The rest of the chain p_{i+2}...p_{n-1} rotates.
            
            # Let's check if p_{i+1} moves.
            # If the axis is the line segment p_i-p_{i+1}, then p_{i+1} is on the axis.
            # So p_{i+1} stays put.
            
            seg_indices = list(range(i + 2, n))
            if not seg_indices:
                continue
                
            new_positions = []
            for idx in seg_indices:
                p = current_chain[idx]
                # Translate to origin at p_{i+1}?
                # No, rotate around the axis line.
                # The axis line passes through p_i and p_{i+1}.
                # Let's define the rotation relative to p_{i+1}.
                # Vector from p_{i+1} to p: w = p - p_{i+1}
                # Rotate w around the axis direction.
                
                w = [p[k] - p1[k] for k in range(3)]
                
                # Identify the two coordinates perpendicular to the axis
                perp_coords = [k for k in range(3) if k != axis_coord]
                c1 = perp_coords[0]
                c2 = perp_coords[1]
                
                # Rotation in c1-c2 plane
                # 90 deg: (x, y) -> (-y, x) or (y, -x)
                # 180 deg: (x, y) -> (-x, -y)
                # 270 deg: (x, y) -> (y, -x)
                
                wc1 = w[c1]
                wc2 = w[c2]
                
                if angle == 1: # 90
                    new_wc1 = -wc2
                    new_wc2 = wc1
                elif angle == 2: # 180
                    new_wc1 = -wc1
                    new_wc2 = -wc2
                else: # 270
                    new_wc1 = wc2
                    new_wc2 = -wc1
                    
                new_p = list(p1)
                new_p[axis_coord] = p1[axis_coord] + w[axis_coord] # w[axis_coord] is 0 for perp? No, w is vector from p1.
                # Wait, the axis is aligned with axis_coord.
                # So the component of w along the axis is w[axis_coord].
                # This component does not change during rotation around the axis.
                # The perpendicular components change.
                
                new_p[axis_coord] = p1[axis_coord] + w[axis_coord]
                new_p[c1] = p1[c1] + new_wc1
                new_p[c2] = p1[c2] + new_wc2
                
                new_positions.append(tuple(new_p))
            
            if _is_valid_move(seg_indices, new_positions):
                new_chain = list(current_chain)
                for idx, pos in zip(seg_indices, new_positions):
                    new_chain[idx] = pos
                
                new_energy = _calculate_energy(res, new_chain)
                delta_e = new_energy - current_energy
                valid = True
        
        if not valid:
            step += 1
            continue
            
        # Accept or reject
        if delta_e < 0:
            # Improvement
            current_chain = new_chain
            current_energy = new_energy
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Worse, accept with probability exp(-delta_e / t)
            # delta_e is positive (energy increase)
            # Energy is negative, so increase means less negative (worse).
            # Delta E = E_new - E_old. If E_new > E_old (e.g. -5 > -10), Delta E > 0.
            # We want to minimize energy.
            # So we accept if rng < exp(-Delta_E / T).
            
            # Note: Energy values are small integers.
            # T starts at 1.0.
            # If Delta_E is 1, prob is exp(-1) ~ 0.36.
            
            if t > 1e-4:
                prob = math.exp(-delta_e / t)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
            
        t *= cooling
        step += 1
        
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multi-restart Beam Search + SA."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Start with 0 (unfolded)
    
    n = len(sequence)
    
    # Determine number of restarts based on time
    # Estimate time per restart.
    # Beam search is fast. SA is the bottleneck.
    
    while _time_remaining() > 1.0:
        # Generate initial chains
        candidates = _beam_search_init(sequence, dimension, width=20, rng=rng)
        
        if not candidates:
            # Fallback to a simple line if beam search fails
            candidates = [[(i, 0, 0) for i in range(n)]]
        
        # Pick a random candidate or the best one from beam
        # Beam search returns sorted by score. Let's pick the best.
        init_chain = candidates[0]
        
        # Run SA
        # Adjust max_steps based on N and time
        # If N is small, more steps. If N is large, fewer steps.
        max_steps = max(500, 5000 // max(1, n // 10))
        
        improved_chain = _simulated_annealing(sequence, init_chain, rng, max_steps=max_steps)
        
        energy = _calculate_energy(sequence, improved_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved_chain
            
        # Check time
        if _time_remaining() < 5.0:
            break

    if best_chain is None:
        # Should not happen, but just in case
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
