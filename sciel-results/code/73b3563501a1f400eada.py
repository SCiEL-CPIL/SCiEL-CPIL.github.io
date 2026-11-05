"""HP lattice protein folding using direct energy-driven search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _build_chain_direct(sequence, rng, temperature=1.0):
    """Build a chain directly by growing, placing H residues to maximize contacts."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0
            
            # Prefer positions that leave more free space
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            
            # Add random perturbation
            score += rng.random() * temperature
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        # Probabilistic choice based on temperature
        if temperature > 0.5 and len(candidates) > 1 and rng.random() < 0.2:
            best_site = candidates[rng.randint(1, min(2, len(candidates) - 1))][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _generate_compact_templates(n, num_templates=50, rng=None):
    """Generate a variety of compact geometric templates (spirals, rectangles)."""
    if rng is None:
        rng = random.Random(42)
        
    templates = []
    seen = set()
    
    # Strategy 1: Standard Spirals (Right and Left turns)
    for turn_dir in [1, -1]: # 1 for right, -1 for left
        for start_dir_idx in range(4):
            start_dirs = [(1,0), (0,1), (-1,0), (0,-1)]
            start_dir = start_dirs[start_dir_idx]
            
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            direction = start_dir
            
            for i in range(1, n):
                found = False
                for _ in range(4):
                    x, y = chain[-1]
                    dx, dy = direction
                    new_pos = (x + dx, y + dy)
                    
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        found = True
                        break
                    
                    # Turn
                    if turn_dir == 1:
                        direction = (dy, -dx)
                    else:
                        direction = (-dy, dx)
                
                if not found:
                    break
            
            if len(chain) == n:
                # Normalize
                min_x = min(p[0] for p in chain)
                min_y = min(p[1] for p in chain)
                normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
                if normalized not in seen:
                    seen.add(normalized)
                    templates.append(list(chain))
                    
    # Strategy 2: Rectangular "Snake" patterns
    # Try to fill a rectangle roughly sqrt(n) x sqrt(n)
    width = int(math.sqrt(n)) + 1
    if width < 2: width = 2
    
    for w in range(max(2, width-2), width+3):
        h = (n + w - 1) // w
        # Generate a snake path in a w x h grid
        chain = []
        occupied = set()
        x, y = 0, 0
        dir_x, dir_y = 1, 0
        
        for i in range(n):
            chain.append((x, y))
            occupied.add((x, y))
            
            # Move
            if dir_x == 1 and (x + 1 >= w or (x+1, y) in occupied):
                if y + 1 < h and (x, y+1) not in occupied:
                    dir_x, dir_y = 0, 1
                else:
                    break
            elif dir_x == -1 and (x - 1 < 0 or (x-1, y) in occupied):
                if y + 1 < h and (x, y+1) not in occupied:
                    dir_x, dir_y = 0, 1
                else:
                    break
            elif dir_y == 1 and (y + 1 >= h or (x, y+1) in occupied):
                if x - 1 >= 0 and (x-1, y) not in occupied:
                    dir_x, dir_y = -1, 0
                else:
                    break
            elif dir_y == -1 and (y - 1 < 0 or (x, y-1) in occupied):
                if x - 1 >= 0 and (x-1, y) not in occupied:
                    dir_x, dir_y = -1, 0
                else:
                    break
            
            x += dir_x
            y += dir_y
            
        if len(chain) == n:
            min_x = min(p[0] for p in chain)
            min_y = min(p[1] for p in chain)
            normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
            if normalized not in seen:
                seen.add(normalized)
                templates.append(list(chain))
                
    return templates

def _refine_chain_fast(sequence, chain, rng, max_iter=2000, deadline=None):
    """Refine a chain using delta-energy calculations for speed.
    Returns (best_chain, best_energy)."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # Precompute H positions for faster delta check? 
    # Actually, just checking neighbors is O(1) per move if we track occupied.
    
    current_energy = best_energy
    current_chain = list(best_chain)
    
    # Simple Hill Climbing with random restarts if stuck
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        # Randomly select a residue to try moving
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try all 4 neighbors
        moves = list(_neighbours(old_pos))
        rng.shuffle(moves)
        
        for new_pos in moves:
            if new_pos in occupied:
                continue
            
            # Check connectivity
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                continue
            
            # Calculate Delta Energy
            # Remove old_pos from consideration, add new_pos
            # Delta = Energy(new) - Energy(old)
            
            delta = 0
            
            # Check contacts lost by moving i from old_pos
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta += 1 # We lose 1 contact (energy goes up by 1, so delta is +1)
            
            # Check contacts gained by moving i to new_pos
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta -= 1 # We gain 1 contact (energy goes down by 1, so delta is -1)
            
            # Note: The above delta calculation is slightly tricky because 
            # 'occupied' contains i at old_pos. 
            # When checking neighbors of old_pos, we exclude i-1 and i+1.
            # When checking neighbors of new_pos, we exclude i-1 and i+1.
            # This is correct because i-1 and i+1 are adjacent to i in the chain,
            # but they are not 'contacts' in the HP model definition (which requires non-adjacent).
            
            new_energy = current_energy + delta
            
            if new_energy < current_energy:
                # Accept move
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                current_energy = new_energy
                improved = True
                break
            elif rng.random() < 0.1: # Small probability of accepting worse move (Simulated Annealing lite)
                # Only accept if significantly better than best so far? No, just random walk
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                current_energy = new_energy
                improved = True
                break
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)
            
        if not improved:
            # If stuck, try a random kick
            if rng.random() < 0.5:
                # Pick a random interior node and move it randomly to an empty spot
                # This helps escape local minima
                for _ in range(5):
                    i = rng.randint(1, n - 2)
                    old_pos = current_chain[i]
                    moves = list(_neighbours(old_pos))
                    rng.shuffle(moves)
                    for new_pos in moves:
                        if new_pos in occupied:
                            continue
                        prev_pos = current_chain[i - 1]
                        next_pos = current_chain[i + 1]
                        if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) == 1 and \
                           abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) == 1:
                            # Perform move regardless of energy
                            current_chain[i] = new_pos
                            del occupied[old_pos]
                            occupied[new_pos] = i
                            current_energy = _energy(sequence, current_chain) # Recalc for safety after kick
                            break
                    else:
                        continue
                    break
                    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    # 210s limit. Use 200s for safety.
    deadline = start_time + 200.0 
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0 # 0 is unfolded (no contacts)
    
    # 1. Generate Templates
    templates = _generate_compact_templates(n, num_templates=100, rng=rng)
    
    # 2. Evaluate Templates
    for template in templates:
        if time.time() > deadline - 20.0:
            break
        # Direct assignment
        energy = _energy(sequence, template)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(template)
            
        # Try reversed sequence on same template? 
        # The template is a geometric path. The sequence is assigned in order.
        # Reversing the sequence means sequence[-1] is at template[0].
        # This is equivalent to traversing the template in reverse.
        # Since the template is a path, traversing it in reverse is just the reverse list of points.
        # But the energy is symmetric? 
        # Energy depends on pairs (i, j). If we reverse the sequence, the pairs are the same spatial locations.
        # So _energy(sequence, template) == _energy(sequence[::-1], template[::-1])
        # But template[::-1] is a valid chain too.
        # However, we just need to check if the reverse sequence on the *same* forward template gives better energy.
        # This is not the same as reversing the template.
        # Let's just stick to direct assignment for templates as they are compact.
        
    # 3. Direct Growth with Refinement
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts:
        if time.time() > deadline - 30.0:
            break
            
        # Vary temperature
        temp = 0.5 + rng.random() * 2.0
        chain = _build_chain_direct(sequence, rng, temperature=temp)
        
        if chain is not None and len(chain) == n:
            attempts += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
                
            # Immediate quick refinement on good candidates
            if energy < -2: # Only refine if it's already decent
                if time.time() < deadline - 20.0:
                    refined_chain, refined_energy = _refine_chain_fast(sequence, chain, rng, max_iter=200, deadline=deadline - 15.0)
                    if refined_energy < best_energy:
                        best_energy = refined_energy
                        best_chain = refined_chain

    # 4. Final Deep Refinement on the Best Chain
    if best_chain is not None and time.time() < deadline - 10.0:
        # Use a large budget for the final polish
        final_chain, final_energy = _refine_chain_fast(sequence, best_chain, rng, max_iter=10000, deadline=deadline - 5.0)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy
            
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
