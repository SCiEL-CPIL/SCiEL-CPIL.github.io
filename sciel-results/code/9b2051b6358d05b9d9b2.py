"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Global time limit management
TIME_LIMIT = 120.0
START_TIME = None
TIME_FRACTION = 0.85 # Stop searching when 85% of time is used to be safe

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _init_time():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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

def _build_state(sequence, chain):
    """Build data structures for fast energy updates."""
    n = len(chain)
    # occupied: map from point to index
    occupied = {p: i for i, p in enumerate(chain)}
    # energy: current total energy (negative contacts)
    # We need to track contacts. 
    # A contact is a pair (i, j) such that |i-j|>1, sequence[i]=='H', sequence[j]=='H' and dist(p_i, p_j)==1.
    # Let's just recalculate energy from scratch if it's fast enough, or maintain a set of contacts.
    # For n=100, recalc is O(N). 100 steps is trivial. 
    # But we do millions of steps. 
    # Let's maintain a set of contact pairs (min(i,j), max(i,j)).
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    energy = -len(contacts)
    return occupied, contacts, energy

def _apply_move(chain, occupied, move_type, idx, new_pos):
    """
    Apply a move to chain and occupied.
    Returns None if invalid.
    move_type: 'local' (single point) or 'pivot' (segment)
    """
    n = len(chain)
    
    if move_type == 'local':
        old_pos = chain[idx]
        if new_pos in occupied:
            # It's only valid if it's the same position (no-op) or we are moving TO a spot that is being vacated?
            # In local move, we move point idx to new_pos.
            # If new_pos is occupied by k, and k != idx, it's invalid unless k is being moved too (not in local).
            if new_pos == old_pos:
                return True, 0 # No change
            return False, 0
        
        # Remove old
        del occupied[old_pos]
        # Add new
        occupied[new_pos] = idx
        # Update chain
        chain[idx] = new_pos
        return True, 1 # 1 point changed

    elif move_type == 'pivot':
        # Pivot at idx. Rotate segment idx+1 to end.
        # We need to check if the new positions collide with the rest of the chain (0 to idx) or each other.
        # Let's generate the new positions for the tail.
        # Rotation around pivot point chain[idx].
        pivot_point = chain[idx]
        # The tail starts at chain[idx+1]
        # We need to rotate the vector (chain[idx+1] - pivot_point) by 90, 180, 270.
        # Actually, we can just try the 3 rotations.
        # The move is defined by: pick idx, pick angle (90, 180, 270).
        # Calculate new tail.
        # If any new point in tail is in occupied (excluding the old tail points) -> Invalid.
        # Note: The old tail points are removed from occupied during the check?
        # Strategy: 
        # 1. Identify old_tail_points = chain[idx+1:]
        # 2. Calculate new_tail_points.
        # 3. Check if any new_tail_point is in occupied AND not in old_tail_points.
        
        old_tail = chain[idx+1:]
        if not old_tail:
            return False, 0
            
        # Determine rotation
        # Vector from pivot to first tail element
        v = tuple(old_tail[0][k] - pivot_point[k] for k in range(len(pivot_point)))
        
        # Rotations in 2D:
        # (x, y) -> (-y, x) 90 deg
        # (x, y) -> (-x, -y) 180 deg
        # (x, y) -> (y, -x) 270 deg
        # We need to pass the angle or the transformation. 
        # Let's assume 'new_pos' in this context is the angle multiplier or the specific rotation type.
        # Let's redefine interface for pivot: move_type='pivot', idx=cut, new_pos=angle (1,2,3 for 90,180,270)
        
        angle = new_pos
        new_tail = []
        # We need to apply the same rotation to all points in the tail relative to the pivot.
        # Since it's a rigid rotation, we can just rotate the vector from pivot to each point.
        for p in old_tail:
            vec = tuple(p[k] - pivot_point[k] for k in range(len(pivot_point)))
            if angle == 1: # 90
                nvec = (-vec[1], vec[0])
            elif angle == 2: # 180
                nvec = (-vec[0], -vec[1])
            elif angle == 3: # 270
                nvec = (vec[1], -vec[0])
            else:
                return False, 0
            np = tuple(pivot_point[k] + nvec[k] for k in range(len(pivot_point)))
            new_tail.append(np)
        
        # Check collisions
        # The new tail must not collide with the head (0..idx)
        # And the new tail must not collide with itself (shouldn't happen for rigid rotation of SAW? 
        # Actually, if the tail was self-avoiding, the rotated tail is also self-avoiding.
        # So we only need to check against the head.
        # Also, we must ensure new_tail doesn't overlap with old_tail if old_tail is still in 'occupied' during check?
        # We check against 'occupied' which currently contains the OLD chain.
        # So if a new point is in occupied, it's a collision UNLESS it is one of the old tail points (which are being replaced).
        # But since the rotation is rigid, new_tail points are distinct from each other.
        # We just need to ensure new_tail points are not in head.
        
        head_set = set(chain[:idx+1])
        for np in new_tail:
            if np in head_set:
                return False, 0
            # Also check if np is in occupied but part of old_tail?
            # If np is in occupied, and np is NOT in head_set, it must be in old_tail.
            # If np is in old_tail, it's fine (it's being moved).
            # If np is NOT in old_tail, it's a collision with head or something else?
            # Since head_set is checked, the only other possibility is collision with old_tail itself?
            # No, new_tail is a permutation of rotated old_tail.
            # Wait, if new_tail[0] == old_tail[0] (180 deg rotation of a point? No, rotation changes position unless vector is 0 which is impossible for SAW).
            # So new_tail points are distinct from old_tail points? Not necessarily. 
            # Example: Pivot (0,0), Tail (1,0). 180 deg -> (-1,0). Distinct.
            # So if np in occupied, it must be in head_set (checked) or in old_tail.
            # If it's in old_tail, it's OK.
            # So we just need to check head_set.
            
        # If valid, apply
        for i, np in enumerate(new_tail):
            old_p = old_tail[i]
            del occupied[old_p]
            occupied[np] = idx + 1 + i
            chain[idx + 1 + i] = np
        return True, len(new_tail)
    return False, 0

def _undo_move(chain, occupied, move_type, idx, new_pos, changed_count):
    """Undo a move."""
    n = len(chain)
    if move_type == 'local':
        if changed_count == 0:
            return
        new_pos = chain[idx]
        # We need the old position. We didn't store it.
        # This is a problem. We need to store the old state or calculate inverse.
        # For local move, the "old" position is not easily recoverable if we don't store it.
        # Better approach: Store the list of (old_pos, new_pos, index) for the move.
        # Or, simpler: just keep a copy of the chain? 
        # For N=100, copying a list of 100 tuples is fast.
        # Let's switch to a simpler SA implementation that uses copies for safety and simplicity, 
        # as Python list copies are O(N) and N is small.
        pass

def _sa_fold(sequence, chain, time_budget):
    """Simulated Annealing optimization."""
    n = len(chain)
    rng = random.Random(42)
    
    # Initial state
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Build occupied set
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Pre-compute H indices for quick check
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    start_time = time.time()
    end_time = start_time + time_budget
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < end_time:
        # Generate move
        # 50% local, 50% pivot
        if rng.random() < 0.5:
            # Local move
            # Pick a random index
            idx = rng.randint(1, n - 2) # Avoid ends to keep it simple? Or allow ends? 
            # Moving ends is fine.
            idx = rng.randint(0, n - 1)
            old_pos = current_chain[idx]
            # Try moving to a neighbor of old_pos
            # Or a random nearby point?
            # Let's try neighbors of old_pos first.
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            moved = False
            for new_pos in neighbors:
                if new_pos in occupied:
                    if new_pos == old_pos:
                        continue
                    # Collision
                    continue
                # Check if moving idx to new_pos causes collision with others?
                # Only idx is moving.
                # So we just need to check if new_pos is free.
                # We already checked `if new_pos in occupied`.
                # But wait, if we move idx, we remove old_pos. 
                # If new_pos is occupied by some other k, it's invalid.
                # So `if new_pos in occupied` is the correct check.
                
                # Apply
                del occupied[old_pos]
                occupied[new_pos] = idx
                current_chain[idx] = new_pos
                
                # Calculate Delta Energy
                # Only neighbors of old_pos and new_pos matter.
                # We need to check H-H contacts involving idx or neighbors of idx.
                # This is complex to do incrementally without a contact set.
                # Given N=100, recalculating full energy is O(N) ~ 100 ops.
                # This is very fast.
                new_energy = _energy(sequence, current_chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
                else:
                    # Undo
                    del occupied[new_pos]
                    occupied[old_pos] = idx
                    current_chain[idx] = old_pos
            
            if not moved:
                # No valid move found or rejected.
                # If rejected, we just continue.
                pass
                
        else:
            # Pivot move
            # Pick cut index
            idx = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            
            # Check validity and apply
            # We need to calculate new tail.
            pivot_point = current_chain[idx]
            old_tail = current_chain[idx+1:]
            
            new_tail = []
            valid = True
            for p in old_tail:
                vec = tuple(p[k] - pivot_point[k] for k in range(2))
                if angle == 1:
                    nvec = (-vec[1], vec[0])
                elif angle == 2:
                    nvec = (-vec[0], -vec[1])
                else:
                    nvec = (vec[1], -vec[0])
                np = tuple(pivot_point[k] + nvec[k] for k in range(2))
                new_tail.append(np)
            
            # Check collision with head (0..idx)
            head_set = set(current_chain[:idx+1])
            for np in new_tail:
                if np in head_set:
                    valid = False
                    break
            
            if valid:
                # Apply
                for i, np in enumerate(new_tail):
                    old_p = old_tail[i]
                    del occupied[old_p]
                    occupied[np] = idx + 1 + i
                    current_chain[idx + 1 + i] = np
                
                new_energy = _energy(sequence, current_chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                else:
                    # Undo
                    for i, old_p in enumerate(old_tail):
                        np = new_tail[i]
                        del occupied[np]
                        occupied[old_p] = idx + 1 + i
                        current_chain[idx + 1 + i] = old_p
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat slightly? Or just continue.
            # Let's do a simple reheat to escape plateaus
            if rng.random() < 0.1:
                T = max(T, 0.5)

    # Final Hill Climb
    # Try to improve best_chain
    current_chain = list(best_chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    improved = True
    while improved and time.time() < end_time:
        improved = False
        # Try local moves on current_chain
        for i in range(n):
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied and new_pos != old_pos:
                    continue
                # Apply
                if new_pos != old_pos:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    current_chain[i] = new_pos
                    new_energy = _energy(sequence, current_chain)
                    if new_energy < current_energy:
                        current_energy = new_energy
                        improved = True
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
                    else:
                        # Undo
                        del occupied[new_pos]
                        occupied[old_pos] = i
                        current_chain[i] = old_pos
                else:
                    # No move
                    pass
        # If no local move improved, try a pivot? 
        # For simplicity, just local hill climb is often enough for the final polish.
        
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Greedy Init + Simulated Annealing."""
    _init_time()
    elapsed = _init_time()
    remaining = TIME_LIMIT - elapsed
    
    if remaining < 1.0:
        # Not enough time, return a straight line
        return [list(p) for p in [(i, 0) for i in range(len(sequence))]]
    
    # Allocate time: 10% for init, 90% for SA
    init_time_budget = min(1.0, remaining * 0.1)
    sa_time_budget = remaining - init_time_budget - 1.0 # 1s safety margin
    
    # 1. Generate a good initial chain using Greedy
    rng = random.Random(12345 + len(sequence))
    best_init_chain = None
    best_init_energy = 0
    
    # Try a few greedy runs to get a good start
    start_t = time.time()
    attempts = 0
    while time.time() - start_t < init_time_budget:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_init_chain is None or energy < best_init_energy:
            best_init_chain = chain
            best_init_energy = energy
        attempts += 1
        # If we found a very good one (e.g. low energy), stop early?
        # For now, just use time budget.
        
    if best_init_chain is None:
        # Fallback: Straight line
        best_init_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_init_energy = _energy(sequence, best_init_chain)

    # 2. Simulated Annealing
    if sa_time_budget > 1.0:
        final_chain = _sa_fold(sequence, best_init_chain, sa_time_budget)
    else:
        final_chain = best_init_chain
        
    return [list(p) for p in final_chain]
# EVOLVE-BLOCK-END
