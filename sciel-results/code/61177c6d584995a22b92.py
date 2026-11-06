"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)

def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _full_energy(sequence, chain):
    """Compute the HP contact energy of a conformation from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initialize_state(sequence, chain):
    """
    Initialize the incremental energy tracking state.
    Returns:
        occupied: set of (x, y)
        h_contacts: set of (i, j) tuples where i < j and they are H-H contacts
        energy: current energy
    """
    n = len(chain)
    occupied = set(chain)
    where = {p: i for i, p in enumerate(chain)}
    h_contacts = set()
    energy = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    h_contacts.add((i, j))
                    energy -= 1
                    
    return occupied, h_contacts, energy

def _apply_move(chain, move_type, params, occupied, h_contacts, sequence):
    """
    Apply a move and return the new chain, updated occupied, updated h_contacts, 
    and energy delta. Returns None if the move is invalid.
    """
    n = len(chain)
    new_chain = chain[:]
    
    if move_type == 'pivot':
        pivot_idx, angle = params
        if pivot_idx < 1 or pivot_idx >= n - 1:
            return None
        
        pivot = chain[pivot_idx]
        # Segment to rotate is from pivot_idx + 1 to end
        seg_start = pivot_idx + 1
        segment = chain[seg_start:]
        
        # Determine new positions for the segment
        new_segment = []
        # We need to check collisions with the fixed part (0..pivot_idx)
        # and internal collisions of the segment (which are preserved by rotation)
        # but we must ensure the new segment doesn't collide with the fixed part.
        
        # Also, the first point of the new segment must be adjacent to the pivot.
        
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else: # 180
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            new_segment.append(new_p)
            
        # Check if first new point is adjacent to pivot
        if abs(new_segment[0][0] - pivot[0]) + abs(new_segment[0][1] - pivot[1]) != 1:
            return None
            
        # Check collisions with fixed part
        fixed_set = set(chain[:pivot_idx + 1])
        for p in new_segment:
            if p in fixed_set:
                return None
                
        # Update chain
        new_chain[seg_start:] = new_segment
        
    elif move_type == 'slither':
        # Move last point
        last_idx = n - 1
        prev = chain[last_idx - 1]
        new_pos = params[0]
        
        if new_pos == chain[last_idx]:
            return None
        if new_pos in occupied and new_pos != chain[last_idx]:
            return None
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return None
            
        new_chain[last_idx] = new_pos
        
    elif move_type == 'crankshaft':
        # Move a segment of length 2 or 3
        # Params: (start_idx, length, dx, dy)
        start_idx, length, dx, dy = params
        end_idx = start_idx + length
        if end_idx > n:
            return None
            
        segment = chain[start_idx:end_idx]
        new_segment = [(x + dx, y + dy) for x, y in segment]
        
        # Check collisions
        # The segment moves as a rigid body.
        # We need to check if any new position is occupied by a node NOT in the segment.
        # And check connectivity with head and tail.
        
        head = chain[:start_idx]
        tail = chain[end_idx:]
        
        # Check connectivity
        if head:
            h_last = head[-1]
            s_first = new_segment[0]
            if abs(h_last[0] - s_first[0]) + abs(h_last[1] - s_first[1]) != 1:
                return None
        if tail:
            t_first = tail[0]
            s_last = new_segment[-1]
            if abs(t_first[0] - s_last[0]) + abs(t_first[1] - s_last[1]) != 1:
                return None
                
        # Check collisions
        # Occupied set currently contains all nodes.
        # We remove the old segment nodes from occupied, check new ones, then restore.
        # But we need to be careful: the new segment nodes might collide with each other?
        # No, rigid body move preserves internal distances.
        # They might collide with head/tail.
        
        # Let's construct the set of nodes that are NOT in the moving segment
        static_nodes = set(head) | set(tail)
        
        for p in new_segment:
            if p in static_nodes:
                return None
            # Also check if p is in the old segment but at a different index?
            # Since it's a rigid move, if p was in old segment, it's just the same node moved.
            # But we must ensure p is not occupied by a static node.
            # The `static_nodes` check covers this.
            
        new_chain[start_idx:end_idx] = new_segment
        
    else:
        return None

    # If we reach here, the move is geometrically valid (no self-intersection).
    # Now we need to calculate the energy delta and update the state.
    
    # To do this efficiently, we identify which contacts are broken and which are formed.
    # This is complex to do incrementally without a full recalculation of contacts for the moved nodes.
    # Given the complexity, and since N is small (<=100), a full recalculation of energy 
    # for the new chain is O(N). 
    # Wait, _full_energy is O(N^2) in worst case if we do naive neighbor checks?
    # No, _full_energy uses a dict lookup, so it's O(N * 4) = O(N).
    # So calculating full energy for the new chain is fast.
    
    new_energy = _full_energy(sequence, new_chain)
    delta = new_energy - chain_energy # chain_energy needs to be passed or tracked
    
    # We don't have chain_energy passed in this helper signature easily without refactoring.
    # Let's refactor: The caller will calculate new_energy and compare.
    # This helper just returns the new_chain.
    
    return new_chain

def _grow_constrained(sequence, rng, bias=1.0):
    """Greedy growth with randomization."""
    n = len(sequence)
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Compactness heuristic
            if len(chain) > 5:
                cx = sum(p[0] for p in chain[-5:]) / 5.0
                cy = sum(p[1] for p in chain[-5:]) / 5.0
                dist = abs(site[0] - cx) + abs(site[1] - cy)
            else:
                dist = 0.0
            
            score = bias * gain - 0.2 * dist + 0.1 * free + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _simulated_annealing(sequence, initial_chain, time_limit, rng):
    """
    Run Simulated Annealing on the given chain.
    """
    n = len(initial_chain)
    current_chain = list(initial_chain)
    current_energy = _full_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.1
    # Estimate number of steps? Let's just run until time limit.
    # We will cool down based on time elapsed.
    
    start_time = time.time()
    elapsed = 0.0
    
    while _time_left() > 5.0 and elapsed < time_limit:
        elapsed = time.time() - start_time
        # Cool down linearly with time
        progress = min(1.0, elapsed / time_limit)
        T = T_start - (T_start - T_end) * progress
        
        # Generate a move
        move_type = rng.random()
        if move_type < 0.6:
            # Pivot
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            params = (pivot_idx, angle)
        elif move_type < 0.8:
            # Slither
            last = current_chain[-1]
            prev = current_chain[-2]
            options = [p for p in _neighbours(prev) if p != last]
            if not options:
                continue
            new_pos = rng.choice(options)
            params = (new_pos,)
        else:
            # Crankshaft (length 2 or 3)
            seg_len = rng.choice([2, 3])
            start_idx = rng.randint(1, n - seg_len - 1)
            # Determine perpendicular direction
            # Get vector of segment
            p1 = current_chain[start_idx]
            p2 = current_chain[start_idx + seg_len - 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Perpendicular vectors
            perp1 = (-dy, dx)
            perp2 = (dy, -dx)
            # Normalize? No, we want unit shift.
            # If segment is horizontal (dy=0, dx!=0), perp is vertical (0, +/-1)
            # If segment is vertical (dx=0, dy!=0), perp is horizontal (+/-1, 0)
            # If diagonal? Not possible in square lattice for length 2/3 straight segments?
            # Wait, crankshaft usually applies to straight segments.
            # If the segment is not straight, this move might be ill-defined or just a general shift.
            # Let's only apply if the segment is straight (all points collinear).
            is_straight = True
            for k in range(start_idx, start_idx + seg_len - 1):
                v1 = (current_chain[k+1][0] - current_chain[k][0], current_chain[k+1][1] - current_chain[k][1])
                if k > start_idx:
                    v_prev = (current_chain[k][0] - current_chain[k-1][0], current_chain[k][1] - current_chain[k-1][1])
                    if v1 != v_prev:
                        is_straight = False
                        break
            
            if not is_straight:
                continue
                
            # Shift by perp1 or perp2
            # We need to ensure the shift is by 1 unit.
            # If dx=1, dy=0, perp1=(0,1), perp2=(0,-1). Correct.
            # If dx=2, dy=0 (length 3), perp1=(0,2)? No, we want to shift the whole segment by 1 unit perpendicular.
            # So we use (0, 1) or (0, -1).
            
            shift1 = (0, 1) if abs(dy) > abs(dx) else (1, 0)
            shift2 = (0, -1) if abs(dy) > abs(dx) else (-1, 0)
            
            # Actually, if dx=0, dy=1, perp is (1,0) and (-1,0).
            # If dx=1, dy=0, perp is (0,1) and (0,-1).
            
            if dx == 0:
                shifts = [(1, 0), (-1, 0)]
            elif dy == 0:
                shifts = [(0, 1), (0, -1)]
            else:
                continue # Not straight or diagonal? Diagonal is not possible for straight SAW segments of len 2/3 on square lattice?
                # Actually, a segment of length 2 can be (0,0)->(1,0). Vector (1,0).
                # A segment of length 3 can be (0,0)->(1,0)->(2,0). Vector (2,0).
                # So dx, dy will be multiples of the step.
                # We just need the perpendicular direction.
                # Let's use the first step direction.
                v = (current_chain[start_idx+1][0] - current_chain[start_idx][0], current_chain[start_idx+1][1] - current_chain[start_idx][1])
                if v[0] != 0:
                    shifts = [(0, 1), (0, -1)]
                else:
                    shifts = [(1, 0), (-1, 0)]
            
            shift = rng.choice(shifts)
            params = (start_idx, seg_len, shift[0], shift[1])
            
        new_chain = _apply_move(current_chain, move_type, params, None, None, sequence)
        if new_chain is None:
            continue
            
        new_energy = _full_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using diverse initial growth + Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial solutions
    initial_chains = []
    num_inits = 20
    for i in range(num_inits):
        if _time_left() < 10.0:
            break
        bias = 0.5 + 0.5 * rng.random()
        chain = _grow_constrained(sequence, rng, bias=bias)
        if chain:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback: simple random walk
        chain = [(0,0)]
        occupied = {(0,0)}
        for i in range(1, n):
            found = False
            for _ in range(100):
                site = rng.choice(list(_neighbours(chain[-1])))
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    found = True
                    break
            if not found:
                break
        if len(chain) == n:
            initial_chains.append(chain)
        else:
            # Last resort
            initial_chains.append([(i, 0) for i in range(n)])
            
    # Phase 2: Run SA on the best few initial chains
    # Sort by energy
    scored_chains = [( _full_energy(sequence, c), c) for c in initial_chains]
    scored_chains.sort(key=lambda x: x[0])
    
    # Take top 5
    top_chains = [c for e, c in scored_chains[:5]]
    
    # Allocate time
    total_time_budget = 180.0
    time_per_sa = total_time_budget / len(top_chains)
    # Leave some margin
    time_per_sa = min(time_per_sa, 30.0)
    
    for chain in top_chains:
        if _time_left() < 5.0:
            break
            
        # Run SA
        # Use a fraction of the time for each
        local_limit = min(time_per_sa, _time_left() - 5.0)
        if local_limit < 1.0:
            continue
            
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, local_limit, rng)
        
        if sa_energy < best_energy or best_chain is None:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Phase 3: Final Polish (Deterministic Hill Climbing)
    if best_chain and _time_left() > 10.0:
        current_chain = list(best_chain)
        current_energy = _full_energy(sequence, current_chain)
        
        # Simple hill climbing with pivot moves
        for _ in range(100): # Max 100 improving steps
            if _time_left() < 5.0:
                break
                
            best_move = None
            best_delta = 0
            
            n = len(current_chain)
            for pivot_idx in range(1, n - 1):
                for angle in (90, -90, 180):
                    new_chain = _apply_move(current_chain, 'pivot', (pivot_idx, angle), None, None, sequence)
                    if new_chain is None:
                        continue
                    new_energy = _full_energy(sequence, new_chain)
                    delta = new_energy - current_energy
                    if delta < best_delta:
                        best_delta = delta
                        best_move = new_chain
                        
            if best_move is None:
                break
                
            current_chain = best_move
            current_energy += best_delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
