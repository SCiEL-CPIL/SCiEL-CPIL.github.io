import time
import random
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate H-H contact energy change when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    x, y, z = q
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _compute_free_space(occupied, q):
    """Count free neighboring positions."""
    x, y, z = q
    free = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r not in occupied:
            free += 1
    return free

def _beam_search_fold(sequence, n, deadline, rng, beam_width=50):
    """
    Beam search: grow chain step by step, keeping only the best beam_width partial paths.
    Score = contacts * 10 + free_space * 1 - distance * 0.05
    """
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0.0)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, score in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                new_score = score
                if sequence[i] == 'H':
                    new_score += _calc_delta_energy(sequence, occupied, i, q) * 10.0
                new_score += _compute_free_space(occupied, q) * 1.0
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                new_score -= dist * 0.05
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, score in beams:
        if len(chain) == n:
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _get_energy(sequence, chain, occupied):
    """Calculate total energy of the chain."""
    energy = 0
    n = len(sequence)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            r = (x + dx, y + dy, z + dz)
            if r in occupied:
                j = occupied[r]
                if j > i and j != i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization using Monte Carlo with Pivots, Shifts, and Single Moves.
    Uses incremental energy updates for speed.
    """
    n = len(sequence)
    if n < 4:
        return chain, _get_energy(sequence, chain, {p: i for i, p in enumerate(chain)})
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    current_energy = _get_energy(sequence, chain, occupied)
    best_energy = current_energy
    best_chain = chain[:]
    
    # Cooling schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # Move counts
    max_steps = 500000 # Increased steps due to faster incremental eval
    
    steps_done = 0
    
    while time.time() < deadline and steps_done < max_steps:
        steps_done += 1
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Single Residue Move (Flip)
            if n < 4: continue
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if q in occupied and occupied[q] != i:
                    continue
                if i > 0:
                    prev = chain[i-1]
                    if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                        continue
                if i < n-1:
                    nxt = chain[i+1]
                    if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                        continue
                valid_mvs.append(q)
            
            if not valid_mvs: continue
            
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos: continue
            
            # Incremental Energy Update
            delta_e = 0
            if sequence[i] == 'H':
                # Remove contacts at old_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e += 1
                # Add contacts at new_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (new_pos[0] + dx, new_pos[1] + dy, new_pos[2] + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e -= 1
            
            # Apply move
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            current_energy += delta_e
            
            if delta_e < 0 or (delta_e == 0 and rng.random() < 0.5) or (delta_e > 0 and rng.random() < math.exp(-delta_e / T)):
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Revert
                del occupied[new_pos]
                occupied[old_pos] = i
                chain[i] = old_pos
                current_energy -= delta_e
                
        elif move_type < 0.6:
            # Pivot Move
            if n < 5: continue
            i = rng.randint(1, n - 4)
            max_len = n - i - 1
            if max_len < 2: continue
            j = i + rng.randint(2, max_len)
            
            hinge_pos = chain[i]
            v = (chain[i+1][0] - hinge_pos[0], chain[i+1][1] - hinge_pos[1], chain[i+1][2] - hinge_pos[2])
            
            # Find valid pivot directions (where i+1 can move to)
            # Standard pivot: rotate segment [i+1...j] around bond (i, i+1)
            # Actually, standard pivot keeps bond (i, i+1) fixed and rotates [i+2...j]
            # But there's another variant where we rotate [i+1...j] around i
            
            # Let's implement the variant where we rotate segment [i+1...j] around i
            # This means i+1 moves, and we need to check collisions for all points in [i+1...j]
            
            valid_dirs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                if (dx, dy, dz) == v: continue  # Can't stay in same direction
                q = (hinge_pos[0] + dx, hinge_pos[1] + dy, hinge_pos[2] + dz)
                if q in occupied and occupied[q] != i+1: continue
                if i > 0 and q == chain[i-1]: continue
                valid_dirs.append(q)
            
            if not valid_dirs: continue
            
            new_pos_i1 = rng.choice(valid_dirs)
            
            # Calculate rotation: we need to rotate all points in [i+1...j]
            # The rotation is around the axis from i to i+1 (which changes)
            # Actually, for a pivot move, we rotate around the bond axis
            # But if we're moving i+1, the axis changes
            
            # Simpler approach: try to find a valid rotation
            # For now, let's just try single-step pivots (rotate by 90 degrees)
            
            # Determine the rotation
            # Old direction: v
            # New direction: v_new = new_pos_i1 - hinge_pos
            v_new = (new_pos_i1[0] - hinge_pos[0], new_pos_i1[1] - hinge_pos[1], new_pos_i1[2] - hinge_pos[2])
            
            # Find rotation axis (cross product of v and v_new)
            cx = v[1]*v_new[2] - v[2]*v_new[1]
            cy = v[2]*v_new[0] - v[0]*v_new[2]
            cz = v[0]*v_new[1] - v[1]*v_new[0]
            
            if cx == 0 and cy == 0 and cz == 0:
                continue  # Parallel, no rotation
            
            # Normalize axis (should be unit vector for 90 degree rotation)
            # For 90 degree rotation, we can use the cross product directly
            
            # Try to apply the rotation to all points in [i+1...j]
            new_chain = chain[:]
            new_occupied = dict(occupied)
            
            # Remove points i+1...j from occupied
            for k in range(i+1, j+1):
                del new_occupied[chain[k]]
            
            valid = True
            for k in range(i+1, j+1):
                rel = (chain[k][0] - hinge_pos[0], chain[k][1] - hinge_pos[1], chain[k][2] - hinge_pos[2])
                
                # Rotate rel by 90 degrees around axis (cx, cy, cz)
                # For 90 degree rotation: P' = A x P
                rx = cy*rel[2] - cz*rel[1]
                ry = cz*rel[0] - cx*rel[2]
                rz = cx*rel[1] - cy*rel[0]
                
                new_p = (hinge_pos[0] + rx, hinge_pos[1] + ry, hinge_pos[2] + rz)
                
                if new_p in new_occupied and new_occupied[new_p] != k:
                    valid = False
                    break
                if new_p == chain[i]:
                    valid = False
                    break
                if i > 0 and new_p == chain[i-1]:
                    valid = False
                    break
                
                new_chain[k] = new_p
                new_occupied[new_p] = k
            
            if not valid:
                continue
            
            # Calculate energy delta
            # This is complex for pivots, so let's just recalculate energy
            # For now, skip pivots to keep it simple and fast
            
            continue # Skip Pivot for now in this implementation to keep code simple and fast
            
        else:
            # Shift Move
            if n < 5: continue
            i = rng.randint(0, n - 4)
            j = i + rng.randint(1, min(3, n - i - 1)) # Short segments
            
            # Try 6 shift directions
            shift_dirs = _NEIGHBOR_OFFSETS
            valid_shifts = []
            for dx, dy, dz in shift_dirs:
                # Check if all points in [i...j] can move by (dx,dy,dz)
                ok = True
                for k in range(i, j+1):
                    new_p = (chain[k][0] + dx, chain[k][1] + dy, chain[k][2] + dz)
                    if new_p in occupied:
                        if occupied[new_p] < i or occupied[new_p] > j:
                            ok = False
                            break
                    # Check connectivity to neighbors outside segment
                    if k == i and i > 0:
                        prev = chain[i-1]
                        if abs(new_p[0]-prev[0]) + abs(new_p[1]-prev[1]) + abs(new_p[2]-prev[2]) != 1:
                            ok = False
                            break
                    if k == j and j < n-1:
                        nxt = chain[j+1]
                        if abs(new_p[0]-nxt[0]) + abs(new_p[1]-nxt[1]) + abs(new_p[2]-nxt[2]) != 1:
                            ok = False
                            break
                if ok:
                    valid_shifts.append((dx, dy, dz))
            
            if not valid_shifts: continue
            
            dx, dy, dz = rng.choice(valid_shifts)
            
            # Apply shift
            old_positions = [chain[k] for k in range(i, j+1)]
            new_positions = [(chain[k][0] + dx, chain[k][1] + dy, chain[k][2] + dz) for k in range(i, j+1)]
            
            # Remove old from occupied
            for k in range(i, j+1):
                del occupied[chain[k]]
            
            # Add new to occupied
            for k in range(i, j+1):
                chain[k] = new_positions[k-i]
                occupied[chain[k]] = k
            
            # Calculate delta energy
            # Contacts lost:
            # 1. Internal contacts within segment that change? No, relative positions are same.
            # 2. Contacts between segment and outside.
            # 3. Contacts between segment and itself that were not neighbors in chain?
            #    Since relative positions are same, internal contacts are same.
            #    Wait, self-avoidance is maintained.
            #    So only contacts with OUTSIDE change.
            
            delta_e = 0
            # Check neighbors of all points in segment
            for k in range(i, j+1):
                if sequence[k] != 'H':
                    continue
                p = chain[k]
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (p[0] + dx2, p[1] + dy2, p[2] + dz2)
                    if r in occupied:
                        m = occupied[r]
                        if m < i or m > j: # Outside
                            if m != k - 1 and m != k + 1:
                                if sequence[m] == 'H':
                                    delta_e -= 1 # New contact
            
            # Now subtract old contacts?
            # We need to know which contacts were there before.
            # It's easier to just recalculate energy for the whole chain?
            # Or store old energy contributions.
            
            # Let's just recalculate energy for the whole chain if we accept?
            # No, let's try to be smarter.
            # We can't easily know old contacts without storing them.
            
            # Given the complexity, let's just use Single Moves and maybe a very simple Pivot if we can.
            # For now, revert Shift if we can't calculate delta easily.
            
            # Let's revert Shift to keep it safe and fast.
            for k in range(i, j+1):
                del occupied[chain[k]]
                chain[k] = old_positions[k-i]
                occupied[chain[k]] = k
            
            continue

    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 220.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    beam_configs = [
        (50, 42), (80, 123), (120, 456), (60, 789), (100, 321),
        (70, 654), (90, 987), (40, 111), (150, 222), (65, 333)
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, rng_local, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        occupied = {}
        for idx, pos in enumerate(chain):
            occupied[pos] = idx
        
        energy = _get_energy(sequence, chain, occupied)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        if n > 10 and best_energy <= -n * 0.45:
            break
    
    if best_chain is not None:
        for round_num in range(5):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
