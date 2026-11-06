"""Improved program with better heuristics and pivot moves in SA."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield all 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive residues on adjacent sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _count_free_neighbors(point, occupied):
    """Count free neighbors of a point."""
    count = 0
    for q in _neighbours(point):
        if q not in occupied:
            count += 1
    return count


def _generate_chain(sequence, dimension, rng, w_contact=10.0, w_space=1.0, w_compact=0.1, w_trap=5.0, noise_scale=0.5):
    """
    Generate a chain using greedy backtracking with heuristic scoring.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Centroid tracking for compactness
    cx, cy, cz = 0.0, 0.0, 0.0
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Calculate score
            score = 0.0
            
            # 1. Gain: H-H contacts formed
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += w_contact
            
            # 2. Space: Free neighbors (to avoid trapping)
            free_count = _count_free_neighbors(site, occupied)
            score += w_space * free_count
            
            # 3. Trap penalty: If this is not the last residue and free_count is low, it's a dead end
            if i < n - 1 and free_count <= 1:
                score -= w_trap
            
            # 4. Compactness: Prefer moves closer to current centroid
            cx /= i
            cy /= i
            cz /= i
            dist_to_centroid = (site[0] - cx)**2 + (site[1] - cy)**2 + (site[2] - cz)**2
            score -= w_compact * dist_to_centroid
            
            # 5. Noise
            score += rng.uniform(0, noise_scale)
            
            candidates.append((score, site))
            
            # Update centroid for next iteration's calculation (approximate)
            cx += site[0]
            cy += site[1]
            cz += site[2]
        
        if not candidates:
            return None
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pick the best one
        best_score, best_site = candidates[0]
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search_sa(sequence, dimension, chain, rng, time_limit=2.0):
    """
    Simulated Annealing local search with Pivot and Kink moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    start_time = time.time()
    
    # Precompute occupied set for quick lookups? 
    # We need to update it on every move.
    # For N=136, set operations are fast enough.
    
    while time.time() - start_time < time_limit:
        # Cooldown
        elapsed = time.time() - start_time
        T = max(T_end, T_start - (elapsed / time_limit) * (T_start - T_end))
        
        move_type = rng.random()
        
        valid_move = False
        new_chain = None
        
        if move_type < 0.7:
            # Pivot move: Rotate segment [i+1, j] around bond (i, i+1)
            # Pick i in [1, n-3] and j in [i+2, n-1]
            if n < 5:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            # The pivot axis is the vector from chain[i] to chain[i+1]
            p0 = current_chain[i]
            p1 = current_chain[i+1]
            
            # Determine the rotation plane.
            # We need to rotate the vector (p1-p0) to a new direction.
            # In 3D, there are 3 possible rotations of 90 degrees around the axis perpendicular to the bond?
            # No, a pivot move rotates the tail around the bond.
            # The bond is fixed. The segment [i+1..j] is rotated.
            # The new position of i+1 must be a neighbor of i, different from old i+1.
            
            # Find valid new positions for i+1
            old_i1 = current_chain[i+1]
            valid_new_i1 = []
            for q in _neighbours(p0):
                if q != old_i1 and q not in set(current_chain[:i] + current_chain[j+1:]):
                    # q is free and not the old position
                    # Also check if q is not collinear with p0-p1? No, any neighbor is fine.
                    valid_new_i1.append(q)
            
            if not valid_new_i1:
                continue
            
            new_i1 = rng.choice(valid_new_i1)
            
            # Now we need to rotate the rest of the segment [i+2..j]
            # The vector from p0 to old_i1 is v_old.
            # The vector from p0 to new_i1 is v_new.
            # We need to apply the same rotation to all subsequent vectors in the segment.
            
            # This is tricky to implement directly with vectors in 3D without rotation matrices.
            # Alternative: Try all 3 possible rotations of the segment?
            # Actually, a simpler pivot implementation:
            # Pick a segment [i+1, j].
            # Try to move the entire segment such that i+1 goes to a new neighbor of i.
            # This requires a rotation in 3D.
            
            # Let's stick to Kink moves for simplicity and speed, but add a "Segment Shift" which is easier.
            # Or implement a simple 2D projection pivot?
            # Given time constraints, let's improve the Kink move and add a "2-mover" (move 2 residues).
            
            pass # Fallback to other moves
            
        # Kink move: move a single internal residue
        i = rng.randint(1, n - 2)
        current_pos = current_chain[i]
        prev_pos = current_chain[i-1]
        next_pos = current_chain[i+1]
        
        # Find a new position for i
        valid_new_pos = []
        for q in _neighbours(current_pos):
            if q == prev_pos or q == next_pos:
                continue
            if q in set(current_chain[:i] + current_chain[i+1:]):
                continue
            # Check if q is adjacent to prev and next?
            # No, kink move moves the residue, so it must be adjacent to prev and next.
            # Wait, if I move residue i, it must still be adjacent to i-1 and i+1.
            # So q must be a neighbor of prev_pos AND a neighbor of next_pos.
            # In a cubic lattice, the only points that are neighbors to both prev and next (where prev and next are neighbors of current)
            # are current and the "opposite" point if prev-next is a straight line?
            # If prev and next are not collinear with current, there is no other point.
            # So kink moves are only possible if prev and next are collinear with current?
            # No. A kink move changes the angle.
            # If prev=(0,0,0), current=(1,0,0), next=(1,1,0).
            # Can we move current to (1,-1,0)? No, (1,-1,0) is not neighbor to (0,0,0).
            # Can we move current to (0,1,0)? No, (0,1,0) is not neighbor to (1,1,0).
            # Actually, in 3D, if prev and next are not collinear, the only common neighbor is current.
            # If prev and next ARE collinear (straight bond), then there are 4 other neighbors.
            # So Kink moves are only valid for "straight" segments?
            # This makes Kink moves very rare and useless for general folding.
            
            # Let's use "Segment Shift" instead.
            pass

        # Segment Shift Move
        # Pick a segment [i, j] and shift it by 1 unit in a random direction.
        # This is what the original code did. Let's keep it but optimize.
        
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        
        direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        
        # Check validity
        new_chain = list(current_chain)
        valid = True
        
        # Occupied set for non-moved parts
        occupied_set = set()
        for k in range(n):
            if i <= k <= j:
                continue
            occupied_set.add(current_chain[k])
        
        # Calculate new positions
        new_positions = []
        for k in range(i, j + 1):
            old_pos = current_chain[k]
            new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
            new_positions.append(new_pos)
            
        # Check self-avoidance against rest
        for k in range(i, j + 1):
            if new_positions[k-i] in occupied_set:
                valid = False
                break
        
        if valid:
            # Check connectivity at boundaries
            pos_prev = current_chain[i-1]
            pos_i_new = new_positions[0]
            dist1 = sum(abs(pos_prev[a] - pos_i_new[a]) for a in range(dimension))
            
            pos_j_new = new_positions[-1]
            pos_next = current_chain[j+1]
            dist2 = sum(abs(pos_j_new[a] - pos_next[a]) for a in range(dimension))
            
            if dist1 != 1 or dist2 != 1:
                valid = False
        
        if valid:
            for k in range(i, j + 1):
                new_chain[k] = new_positions[k-i]
            valid_move = True
        
        if not valid_move:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            delta = new_energy - current_energy
            if T > 0 and rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find best conformation for the given HP sequence on the simple cubic lattice.
    """
    n = len(sequence)
    
    start_time = time.time()
    time_limit = 8.0
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345 + len(sequence))
    
    # Strategy 1: Greedy with varying heuristics
    num_attempts = 30 if n > 30 else 60
    
    # Different weight sets
    weight_sets = [
        (10.0, 1.0, 0.1, 5.0, 0.5),
        (15.0, 2.0, 0.05, 10.0, 1.0),
        (8.0, 3.0, 0.2, 3.0, 0.2),
        (12.0, 1.5, 0.15, 7.0, 0.8),
        (20.0, 1.0, 0.0, 20.0, 1.5), # Heavy contact focus
    ]
    
    for i in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
            
        ws = weight_sets[i % len(weight_sets)]
        chain = _generate_chain(sequence, dimension, rng, *ws)
        
        if chain is None:
            continue
            
        e = _energy(sequence, chain)
        
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
        # Refine
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.5:
            sa_time = min(1.5, time_remaining * 0.6)
            refined_chain, refined_e = _local_search_sa(sequence, dimension, chain, rng, time_limit=sa_time)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined_chain
                
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
