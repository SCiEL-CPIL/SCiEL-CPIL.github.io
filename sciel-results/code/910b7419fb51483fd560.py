"""HP lattice protein folding - Monte Carlo Chain with pivot moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 195.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    return [(point[0] + dx, point[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _pivot_move(chain, pos_map, pivot_idx, rng):
    """
    Attempt a pivot move at pivot_idx.
    Returns (success, new_chain, new_pos_map, new_energy_delta)
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n - 1:
        return False, None, None, 0
    
    # Identify the segment to move: [pivot_idx, n-1]
    # The pivot is at pivot_idx. The bond is between pivot_idx-1 and pivot_idx.
    # We rotate the segment [pivot_idx, n-1] around the point chain[pivot_idx-1].
    
    anchor = chain[pivot_idx - 1]
    # Direction from anchor to pivot
    dx = chain[pivot_idx][0] - anchor[0]
    dy = chain[pivot_idx][1] - anchor[1]
    
    # Perpendicular directions
    # If dx=1, dy=0 (Right), perps are Up (0,1) and Down (0,-1)
    # If dx=0, dy=1 (Up), perps are Left (-1,0) and Right (1,0)
    perps = []
    if dx == 1:
        perps = [(0, 1), (0, -1)]
    elif dx == -1:
        perps = [(0, 1), (0, -1)]
    elif dy == 1:
        perps = [(-1, 0), (1, 0)]
    elif dy == -1:
        perps = [(-1, 0), (1, 0)]
    
    if not perps:
        return False, None, None, 0
        
    # Choose a random perpendicular direction
    perp = rng.choice(perps)
    pdx, pdy = perp
    
    # We need to check if the rotated segment fits.
    # The segment is chain[pivot_idx : n].
    # We will compute the new positions.
    # Note: The relative geometry of the segment is preserved, but it is rotated.
    # Actually, a pivot move rotates the segment in the plane.
    # The new direction of the first bond (anchor->new_pivot) is perp.
    # The rest of the segment follows the same relative turns.
    
    # Let's trace the new segment.
    # new_chain_segment = []
    # current_pos = anchor
    # current_dir = perp (this is the direction from anchor to new_pivot)
    
    # We need to replay the moves of the original segment, but rotated.
    # Original segment moves:
    # move 0: chain[pivot_idx] - chain[pivot_idx-1] (this is the bond we are rotating)
    # move k: chain[pivot_idx+k] - chain[pivot_idx+k-1] for k=1..n-pivot_idx-1
    
    # The rotation is a 90 degree turn.
    # If we rotate the entire vector space of the segment by 90 degrees,
    # the relative moves are rotated too.
    
    # Let's define a helper to rotate a vector (x, y) by 90 degrees counter-clockwise.
    # (x, y) -> (-y, x)
    # Or clockwise: (x, y) -> (y, -x)
    
    # The direction from anchor to pivot is (dx, dy).
    # The new direction is (pdx, pdy).
    # This corresponds to a specific rotation.
    # If (dx,dy) is (1,0) and (pdx,pdy) is (0,1), it's CCW.
    # If (dx,dy) is (1,0) and (pdx,pdy) is (0,-1), it's CW.
    
    is_ccw = False
    if (dx, dy) == (1, 0) and (pdx, pdy) == (0, 1): is_ccw = True
    elif (dx, dy) == (-1, 0) and (pdx, pdy) == (0, -1): is_ccw = True
    elif (dx, dy) == (0, 1) and (pdx, pdy) == (-1, 0): is_ccw = True
    elif (dx, dy) == (0, -1) and (pdx, pdy) == (1, 0): is_ccw = True
    else:
        is_ccw = False # It's CW
        
    def rotate_move(mx, my):
        if is_ccw:
            return (-my, mx)
        else:
            return (my, -mx)
            
    new_segment = []
    current_pos = anchor
    new_segment.append(current_pos)
    
    # The first move is the one from anchor to pivot, which is now (pdx, pdy).
    # But wait, the segment is [pivot_idx, n-1].
    # The bond connecting the fixed part to the moving part is (pivot_idx-1) -> (pivot_idx).
    # So the first point of the moving part is chain[pivot_idx].
    # Its new position is anchor + (pdx, pdy).
    
    new_pos = (anchor[0] + pdx, anchor[1] + pdy)
    new_segment[0] = new_pos # This is the new position of chain[pivot_idx]
    
    # Now we need to place the rest of the segment: chain[pivot_idx+1] ... chain[n-1]
    # We replay the moves from the original chain, rotated.
    
    # Original moves within the segment:
    # move 1: chain[pivot_idx+1] - chain[pivot_idx]
    # move k: chain[pivot_idx+k] - chain[pivot_idx+k-1]
    
    valid = True
    # We need to check collisions.
    # The fixed part is chain[0 ... pivot_idx-1].
    # The moving part is new_segment.
    # We must ensure new_segment does not intersect chain[0 ... pivot_idx-1].
    # And new_segment must be self-avoiding (which it is, as it's a rotation of a self-avoiding path).
    # Wait, is a rotation of a self-avoiding path always self-avoiding? Yes, rotation is an isometry.
    
    # So we only need to check collisions with the fixed part.
    
    # Let's build the full new chain to check collisions.
    # fixed_part = chain[0 : pivot_idx]
    # moving_part = new_segment (starting from new position of pivot_idx)
    
    # Let's trace the full moving part.
    cur = new_pos
    for k in range(1, n - pivot_idx):
        # Get original move: chain[pivot_idx + k] - chain[pivot_idx + k - 1]
        ox = chain[pivot_idx + k][0] - chain[pivot_idx + k - 1][0]
        oy = chain[pivot_idx + k][1] - chain[pivot_idx + k - 1][1]
        # Rotate it
        rx, ry = rotate_move(ox, oy)
        cur = (cur[0] + rx, cur[1] + ry)
        new_segment.append(cur)
        
        # Check collision with fixed part
        # We can use a set for fixed part for O(1) lookup
        # But we don't have the set handy here without passing it.
        # Let's pass pos_map or build a set.
        
    # Actually, let's just check collisions in the main loop or return the segment and check outside.
    # For efficiency, let's check collisions here.
    
    # We need to check if any point in new_segment is in chain[0 : pivot_idx].
    # Let's create a set of fixed points.
    # This is O(N) per move, which might be slow.
    # Better to use pos_map.
    
    # pos_map maps point -> index.
    # If a point in new_segment is in pos_map, and its index is < pivot_idx, it's a collision.
    # If its index is >= pivot_idx, it's part of the moving segment (which is allowed to be there? No, we are moving it, so the old positions are vacated).
    # Wait, the old positions of the moving segment are chain[pivot_idx ... n-1].
    # These positions are vacated.
    # So if new_segment contains a point that was in chain[pivot_idx ... n-1], is that a collision?
    # No, because that point is being vacated by the old segment and occupied by the new segment?
    # Actually, if the new segment overlaps with the *old* position of the moving segment, that's fine, as long as it doesn't overlap with the *fixed* part.
    # However, if the new segment overlaps with itself? No, rotation preserves self-avoidance.
    
    # So we only need to check:
    # 1. Does new_segment intersect fixed_part (indices 0 to pivot_idx-1)?
    # 2. Does new_segment intersect the *vacated* space? No, vacated space is free.
    
    # Let's check collisions with fixed part using pos_map.
    for p in new_segment:
        if p in pos_map:
            idx = pos_map[p]
            if idx < pivot_idx:
                valid = False
                break
            # If idx >= pivot_idx, it's an old position of the moving part. That's okay.
            
    if not valid:
        return False, None, None, 0
        
    # Calculate energy delta.
    # This is the tricky part.
    # We need to count H-H contacts.
    # Contacts are between non-adjacent residues.
    # Moving the segment changes:
    # 1. Contacts between moving segment and fixed part.
    # 2. Contacts within the moving segment? No, relative geometry is preserved, so internal contacts are unchanged.
    # 3. The contact at the pivot bond? The bond (pivot_idx-1, pivot_idx) is not a contact (adjacent).
    
    # So we only need to calculate the change in contacts between the moving segment and the fixed part.
    
    # Let's identify H residues in the moving segment.
    # And H residues in the fixed part.
    
    # Old contacts:
    # For each H in moving segment (index i >= pivot_idx), check neighbors.
    # If neighbor is in fixed part (index j < pivot_idx) and is H, it's a contact.
    
    # New contacts:
    # For each H in new moving segment (index i >= pivot_idx), check neighbors.
    # If neighbor is in fixed part (index j < pivot_idx) and is H, it's a contact.
    
    # We need to map the new positions to indices.
    # The indices of the moving segment are unchanged (pivot_idx to n-1).
    
    old_contacts = 0
    for k in range(pivot_idx, n):
        if sequence[k] == 'H':
            old_p = chain[k]
            for q in _neighbours(old_p):
                j = pos_map.get(q)
                if j is not None and j < pivot_idx and sequence[j] == 'H':
                    # Check if j and k are not adjacent.
                    # Since j < pivot_idx and k >= pivot_idx, |j-k| >= 1.
                    # They are adjacent only if j = k-1.
                    # If k == pivot_idx, j = pivot_idx-1. This is the bond. Not a contact.
                    if abs(j - k) > 1:
                        old_contacts += 1
                        
    new_contacts = 0
    # We need to check neighbors of new positions.
    # The fixed part is unchanged.
    for k in range(pivot_idx, n):
        if sequence[k] == 'H':
            new_p = new_segment[k - pivot_idx]
            for q in _neighbours(new_p):
                j = pos_map.get(q)
                if j is not None and j < pivot_idx and sequence[j] == 'H':
                    if abs(j - k) > 1:
                        new_contacts += 1
                        
    delta = new_contacts - old_contacts
    # Energy is -contacts. So delta_energy = -delta_contacts.
    # We want to minimize energy, so maximize contacts.
    # If delta_contacts > 0, energy decreases (good).
    
    # Update chain and pos_map
    # Remove old positions
    for k in range(pivot_idx, n):
        old_p = chain[k]
        # Only remove if it's not also a new position?
        # Actually, we can just delete and re-add.
        # But if a new position is the same as an old position, we delete then add.
        # That's fine.
        del pos_map[old_p]
        
    # Add new positions
    for k in range(pivot_idx, n):
        new_p = new_segment[k - pivot_idx]
        pos_map[new_p] = k
        chain[k] = new_p
        
    return True, chain, pos_map, -delta # Return energy delta


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence) * 7)

    # Initial conformation: random walk or straight line?
    # Straight line is valid but bad energy.
    # Let's try to generate a valid random walk.
    # Or just start with a straight line and let MCC fold it.
    # A straight line is a valid self-avoiding walk.
    
    chain = [(i, 0) for i in range(n)]
    pos_map = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Simulated Annealing / Monte Carlo
    # Start temperature high, cool down.
    # Or just Metropolis with a fixed temperature?
    # Given the time budget, SA is better for escaping minima.
    
    # Determine number of steps based on time.
    # We have ~195s.
    # Let's run until deadline.
    
    # Initial temperature
    T = 5.0
    cooling = 0.999
    
    # To make it faster, we can try multiple restarts if we have time.
    # But MCC is powerful. Let's stick to one long run first.
    # If we have time left, we can restart.
    
    steps = 0
    while _remaining() > 0.5:
        # Choose a random pivot
        pivot_idx = rng.randint(1, n - 2)
        
        # Try the pivot move
        success, new_chain, new_pos_map, delta_E = _pivot_move(chain, pos_map, pivot_idx, rng)
        
        if success:
            # Metropolis criterion
            if delta_E < 0:
                # Accept (energy decreased)
                current_energy += delta_E
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Accept with probability exp(-delta_E / T)
                # delta_E is positive (energy increase)
                p = math.exp(-delta_E / T)
                if rng.random() < p:
                    current_energy += delta_E
                    # Not better than best, but accepted for exploration
            
            # Cool down
            T *= cooling
            if T < 0.1:
                T = 0.1
                
            steps += 1
            
            # Optional: check time more frequently for long sequences
            # The _remaining() check in the while loop is sufficient.

    # Final hill climb to polish?
    # MCC is already quite good. A quick local search might help.
    # Let's do a few single-residue moves if time permits.
    if _remaining() > 2.0:
        # Simple hill climb on best_chain
        hc_chain = list(best_chain)
        hc_pos_map = {p: i for i, p in enumerate(hc_chain)}
        hc_energy = best_energy
        improved = True
        while improved and _remaining() > 1.0:
            improved = False
            for i in range(1, n - 1):
                p = hc_chain[i]
                for q in _neighbours(p):
                    if q == hc_chain[i - 1] or q == hc_chain[i + 1]:
                        continue
                    if q in hc_pos_map and hc_pos_map[q] != i:
                        continue
                    prev = hc_chain[i - 1]
                    nxt = hc_chain[i + 1]
                    if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                        continue
                    if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                        continue
                    # Check collision
                    if q in hc_pos_map and hc_pos_map[q] != i:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    if sequence[i] == 'H':
                        for old_q in _neighbours(p):
                            j = hc_pos_map.get(old_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta += 1
                        for new_q in _neighbours(q):
                            j = hc_pos_map.get(new_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta -= 1
                    # Note: delta here is change in contacts.
                    # If delta > 0, we gain contacts (energy drops).
                    if delta > 0:
                        # Apply move
                        del hc_pos_map[p]
                        hc_chain[i] = q
                        hc_pos_map[q] = i
                        hc_energy -= delta # Energy decreases
                        improved = True
                        if hc_energy < best_energy:
                            best_energy = hc_energy
                            best_chain = list(hc_chain)
                        break # Restart scan after improvement

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
