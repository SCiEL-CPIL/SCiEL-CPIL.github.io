"""HP lattice protein folding using CSP with backjumping and local refinement."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_csp(sequence, dimension, time_limit=10.0, seed=42):
    n = len(sequence)
    start_time = time.time()
    random.seed(seed)
    
    best_chain = None
    best_energy = 0  # We want most negative (most contacts)
    
    # Symmetry breaking: 
    # Residue 0 at (0,0)
    # Residue 1 at (1,0) (if n > 1)
    
    def _search(chain, occupied, current_energy):
        nonlocal best_chain, best_energy, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        i = len(chain)
        if i == n:
            # Complete chain found
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return
            
        last_pos = chain[-1]
        current_residue = sequence[i]
        
        # Generate candidates
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Forward Checking: Does the next residue have a valid move?
            if i < n - 1:
                next_has_move = False
                for next_site in _neighbours(site):
                    if next_site not in occupied and next_site != last_pos:
                        next_has_move = True
                        break
                if not next_has_move:
                    continue

            # Heuristic Scoring
            score = 0.0
            
            # 1. Immediate Contact Gain (High Priority)
            if current_residue == 'H':
                h_contacts = 0
                total_occupied_neighbors = 0
                for q in _neighbours(site):
                    if q in occupied:
                        total_occupied_neighbors += 1
                        j = occupied[q]
                        if sequence[j] == 'H':
                            h_contacts += 1
                # Strong reward for H-H contacts
                score += h_contacts * 100.0
                # Small reward for any contact (packing)
                score += total_occupied_neighbors * 10.0
            
            # 2. Free Space / Trap Avoidance
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            
            # If placing here leaves 0 free neighbors for the next residue (and i < n-1), it's a dead end
            if free_count == 0 and i < n - 1:
                continue # Prune immediately
            
            # Penalize low free space to avoid trapping the chain
            # We want at least 2 free neighbors if possible
            if free_count == 1:
                score -= 50.0
            elif free_count == 2:
                score -= 10.0
            
            # 3. Compactness: Prefer moves that keep the chain close to the origin
            # This encourages compact folding, increasing probability of future contacts
            dist = abs(site[0]) + abs(site[1])
            score -= dist * 5.0
            
            # 4. Potential Contact Density Heuristic
            # If we are placing an H, prefer sites that are "surrounded" by the chain (high occupied neighbors)
            # as this suggests it's in the core.
            if current_residue == 'H':
                # Count occupied neighbors again (we have total_occupied_neighbors if we calculated above, but logic was inside if)
                # Let's just use the distance to centroid or similar.
                pass # The compactness term covers this roughly.
            
            # 5. Randomness for diversity in restarts
            score += random.random() * 10.0
            
            candidates.append((score, site))
            
        if not candidates:
            return
            
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            # Try placing at site
            chain.append(site)
            occupied[site] = i
            
            # Calculate new energy increment
            delta_energy = 0
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (improves) by 1 per contact
            
            _search(chain, occupied, current_energy + delta_energy)
            
            # Backtrack
            chain.pop()
            del occupied[site]
            
            if time.time() - start_time > time_limit:
                return

    # Run search
    init_chain = [(0, 0)]
    init_occupied = {(0, 0): 0}
    
    if n > 1:
        init_chain.append((1, 0))
        init_occupied[(1, 0)] = 1
        _search(init_chain, init_occupied, 0)
    else:
        best_chain = init_chain[:]
        best_energy = 0

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    return best_chain, best_energy

def _local_refine(sequence, chain, time_limit=2.0, seed=42):
    """Improve the chain using local moves (pivots)."""
    n = len(sequence)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    start_time = time.time()
    random.seed(seed)
    current_chain = [tuple(p) for p in chain]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # Temperature for Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random pivot point
        # i is the index of the residue that will be the pivot (stays fixed)
        # j is the index where the subchain is attached (i+1 is the first moved)
        # We pick a segment [i+1, j] to rotate/reflect
        if random.random() < 0.5:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
        else:
            # Single move
            i = random.randint(0, n - 2)
            j = i + 1
            
        # Extract the segment
        pivot = current_chain[i]
        segment = current_chain[i+1:j+1]
        
        # Try all 4 rotations (0, 90, 180, 270) and 2 reflections (x, y)
        # Actually, on a square lattice, we can try moving the whole segment
        # by rotating it around the pivot.
        
        # Get relative coordinates
        rel_coords = [(p[0] - pivot[0], p[1] - pivot[1]) for p in segment]
        
        # Generate possible transformations
        transforms = []
        # Identity
        transforms.append([(x, y) for x, y in rel_coords])
        # Rotate 90
        transforms.append([(-y, x) for x, y in rel_coords])
        # Rotate 180
        transforms.append([(-x, -y) for x, y in rel_coords])
        # Rotate 270
        transforms.append([(y, -x) for x, y in rel_coords])
        
        # Reflections (flip x, flip y, flip both)
        transforms.append([(x, -y) for x, y in rel_coords])
        transforms.append([(-x, y) for x, y in rel_coords])
        transforms.append([(-x, -y) for x, y in rel_coords]) # Same as 180 rot
        
        valid_moves = []
        for trans in transforms:
            new_segment = [(pivot[0] + x, pivot[1] + y) for x, y in trans]
            
            # Check self-avoidance
            # The segment must not intersect the rest of the chain (excluding the pivot and the segment itself)
            # And the segment itself must be self-avoiding (it is, as it's a rotation of a valid segment)
            
            # Create a set of all positions in the new chain
            # To be efficient, we only check intersections with the "fixed" parts
            # Fixed parts: 0..i-1 and j+1..n-1
            fixed_positions = set()
            for k in range(i):
                fixed_positions.add(current_chain[k])
            for k in range(j+1, n):
                fixed_positions.add(current_chain[k])
            
            # Check if any position in new_segment is in fixed_positions
            # Note: pivot is in fixed_positions? No, pivot is index i.
            # The segment starts at i+1.
            # So we check new_segment against fixed_positions (which excludes pivot i).
            
            collision = False
            for p in new_segment:
                if p in fixed_positions:
                    collision = True
                    break
            
            if not collision:
                # Construct new chain
                new_chain = current_chain[:i+1] + new_segment + current_chain[j+1:]
                # Check if the new segment connects to the pivot?
                # The first element of new_segment must be a neighbor of pivot.
                # The last element of new_segment must be a neighbor of current_chain[j+1] (if j < n-1)
                
                # Check connection at start
                first_new = new_segment[0]
                if abs(first_new[0] - pivot[0]) + abs(first_new[1] - pivot[1]) != 1:
                    continue
                    
                # Check connection at end
                if j < n - 1:
                    last_new = new_segment[-1]
                    next_fixed = current_chain[j+1]
                    if abs(last_new[0] - next_fixed[0]) + abs(last_new[1] - next_fixed[1]) != 1:
                        continue
                
                # Calculate energy delta
                # We need to recompute energy or calculate delta.
                # Recomputing is O(N), which is fast enough for N=100.
                new_energy = _compute_energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                valid_moves.append((delta_e, new_chain))
        
        if not valid_moves:
            T *= cooling_rate
            continue
            
        # Pick a move (Greedy or SA)
        if random.random() < 0.8:
            # Greedy: pick best
            best_move = min(valid_moves, key=lambda x: x[0])
            delta_e, new_chain = best_move
            if delta_e <= 0:
                current_chain = new_chain
                current_energy = best_energy if False else new_chain and _compute_energy(sequence, new_chain) # Recalculate to be safe
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                # SA acceptance
                if random.random() < math.exp(-delta_e / T):
                    current_chain = new_chain
                    current_energy = _compute_energy(sequence, new_chain)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
        else:
            # Random valid move
            delta_e, new_chain = random.choice(valid_moves)
            if delta_e <= 0:
                current_chain = new_chain
                current_energy = _compute_energy(sequence, new_chain)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                if random.random() < math.exp(-delta_e / T):
                    current_chain = new_chain
                    current_energy = _compute_energy(sequence, new_chain)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                        
        T *= cooling_rate

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence using CSP-inspired search with local refinement."""
    n = len(sequence)
    
    # Time budget per sequence
    if n <= 20:
        time_limit_csp = 5.0
        time_limit_refine = 5.0
    elif n <= 40:
        time_limit_csp = 4.0
        time_limit_refine = 4.0
    elif n <= 60:
        time_limit_csp = 3.0
        time_limit_refine = 3.0
    else:
        time_limit_csp = 2.0
        time_limit_refine = 2.0
        
    # Multi-restart strategy to explore different parts of the search space
    num_restarts = 3
    time_per_restart = time_limit_csp / num_restarts
    
    best_chain = None
    best_energy = 0
    global_start = time.time()
    
    # Phase 1: CSP Search
    for i in range(num_restarts):
        seed = 42 + i
        chain, energy = _fold_csp(sequence, dimension, time_limit=time_per_restart, seed=seed)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Check if we are running out of total time
        if time.time() - global_start > time_limit_csp * 0.9:
            break
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    # Phase 2: Local Refinement
    if time.time() - global_start < time_limit_csp + time_limit_refine - 0.5:
        remaining_time = time_limit_csp + time_limit_refine - (time.time() - global_start)
        if remaining_time > 0.5:
            refined_chain, refined_energy = _local_refine(sequence, best_chain, time_limit=remaining_time, seed=123)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
