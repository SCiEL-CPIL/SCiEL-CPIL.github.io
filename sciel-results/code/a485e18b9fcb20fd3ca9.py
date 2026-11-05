"""HP lattice protein folding using Hamiltonian path templates with greedy assignment and local search."""
import time
import random
import math
import sys

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, occupied, i, old_pos, new_pos):
    """Calculate the change in energy when moving residue i from old_pos to new_pos."""
    delta = 0
    if sequence[i] == 'H':
        # Contacts lost by moving away from old_pos
        for q in _neighbours(old_pos):
            if q in occupied:
                j = occupied[q]
                if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta += 1
        # Contacts gained by moving to new_pos
        for q in _neighbours(new_pos):
            if q in occupied:
                j = occupied[q]
                if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta -= 1
    return delta

def _generate_hamiltonian_templates(n, rng, num_templates=20):
    """Generate Hamiltonian path templates (self-avoiding walks) of length n."""
    templates = []
    seen = set()
    
    # Strategy 1: Snake patterns in rectangles
    for w in range(2, min(n, 20) + 1):
        h = (n + w - 1) // w
        if w * h < n or w * h > n + w:  # Skip if too far off
            continue
        
        # Try both orientations and starting corners
        for start_x in [0, w - 1]:
            for start_y in [0, h - 1]:
                for dir_x in [1, -1]:
                    for dir_y in [1, -1]:
                        chain = []
                        occupied = set()
                        x, y = start_x, start_y
                        
                        # Determine initial direction based on start position
                        if start_x == 0:
                            init_dx = 1
                        else:
                            init_dx = -1
                        if start_y == 0:
                            init_dy = 1
                        else:
                            init_dy = -1
                        
                        dx, dy = init_dx, 0
                        
                        for i in range(n):
                            if x < 0 or x >= w or y < 0 or y >= h or (x, y) in occupied:
                                break
                            chain.append((x, y))
                            occupied.add((x, y))
                            
                            # Determine next move
                            if dx != 0:
                                # Moving horizontally
                                next_x = x + dx
                                if 0 <= next_x < w and (next_x, y) not in occupied:
                                    x = next_x
                                else:
                                    # Turn vertically
                                    if dy == 0:
                                        dy = 1 if start_y == 0 else -1
                                    # Check if we can move vertically
                                    next_y = y + dy
                                    if 0 <= next_y < h and (x, next_y) not in occupied:
                                        y = next_y
                                        dx, dy = 0, dy
                                    else:
                                        # Try other vertical direction
                                        dy = -dy
                                        next_y = y + dy
                                        if 0 <= next_y < h and (x, next_y) not in occupied:
                                            y = next_y
                                            dx, dy = 0, dy
                                        else:
                                            break
                            else:
                                # Moving vertically
                                next_y = y + dy
                                if 0 <= next_y < h and (x, next_y) not in occupied:
                                    y = next_y
                                else:
                                    # Turn horizontally
                                    if dx == 0:
                                        dx = 1 if start_x == 0 else -1
                                    next_x = x + dx
                                    if 0 <= next_x < w and (next_x, y) not in occupied:
                                        x = next_x
                                        dx, dy = dx, 0
                                    else:
                                        # Try other horizontal direction
                                        dx = -dx
                                        next_x = x + dx
                                        if 0 <= next_x < w and (next_x, y) not in occupied:
                                            x = next_x
                                            dx, dy = dx, 0
                                        else:
                                            break
                        
                        if len(chain) == n:
                            # Normalize
                            min_x = min(p[0] for p in chain)
                            min_y = min(p[1] for p in chain)
                            normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
                            if normalized not in seen:
                                seen.add(normalized)
                                templates.append(list(chain))
                                if len(templates) >= num_templates:
                                    return templates
    
    # Strategy 2: Spiral patterns
    for turn_dir in [1, -1]:
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
                min_x = min(p[0] for p in chain)
                min_y = min(p[1] for p in chain)
                normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
                if normalized not in seen:
                    seen.add(normalized)
                    templates.append(list(chain))
                    if len(templates) >= num_templates:
                        return templates
    
    return templates

def _greedy_assign(sequence, template, rng):
    """Try to assign the sequence to the template, possibly reversing."""
    # Direct assignment
    energy1 = _energy(sequence, template)
    # Reversed sequence on reversed template
    energy2 = _energy(sequence[::-1], template[::-1])
    
    if energy1 <= energy2:
        return template, energy1
    else:
        return template[::-1], energy2

def _local_search(sequence, chain, rng, max_iter=5000, deadline=None):
    """Local search with single-residue moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    current_chain = list(best_chain)
    current_energy = best_energy
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        moves = _neighbours(old_pos)
        rng.shuffle(moves)
        
        for new_pos in moves:
            if new_pos in occupied:
                continue
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                continue
            
            delta = _delta_energy(sequence, current_chain, occupied, i, old_pos, new_pos)
            new_energy = current_energy + delta
            
            if new_energy < current_energy:
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
            elif rng.random() < 0.05:
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                current_energy = new_energy
                break
    
    return best_chain, best_energy

def _two_point_search(sequence, chain, rng, max_iter=2000, deadline=None):
    """Two-point move search: swap two non-adjacent interior residues."""
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    current_chain = list(best_chain)
    current_energy = best_energy
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i = rng.randint(1, n - 2)
        j = rng.randint(1, n - 2)
        if abs(i - j) <= 2:
            continue
        
        old_i = current_chain[i]
        old_j = current_chain[j]
        
        # Try swapping
        new_i = old_j
        new_j = old_i
        
        # Check validity
        prev_i = current_chain[i - 1]
        next_i = current_chain[i + 1]
        prev_j = current_chain[j - 1]
        next_j = current_chain[j + 1]
        
        if new_i == new_j:
            continue
        
        # Check if new positions are valid (not occupied by other residues)
        # Temporarily remove i and j from occupied
        occupied_temp = dict(occupied)
        del occupied_temp[old_i]
        del occupied_temp[old_j]
        
        if new_i in occupied_temp or new_j in occupied_temp:
            continue
        
        # Check connectivity
        if abs(new_i[0] - prev_i[0]) + abs(new_i[1] - prev_i[1]) != 1:
            continue
        if abs(new_i[0] - next_i[0]) + abs(new_i[1] - next_i[1]) != 1:
            continue
        if abs(new_j[0] - prev_j[0]) + abs(new_j[1] - prev_j[1]) != 1:
            continue
        if abs(new_j[0] - next_j[0]) + abs(new_j[1] - next_j[1]) != 1:
            continue
        
        # Calculate energy change
        delta = 0
        if sequence[i] == 'H':
            for q in _neighbours(old_i):
                if q in occupied_temp:
                    k = occupied_temp[q]
                    if k != i - 1 and k != i + 1 and k != j and sequence[k] == 'H':
                        delta += 1
            for q in _neighbours(new_i):
                if q in occupied_temp:
                    k = occupied_temp[q]
                    if k != i - 1 and k != i + 1 and k != j and sequence[k] == 'H':
                        delta -= 1
        if sequence[j] == 'H':
            for q in _neighbours(old_j):
                if q in occupied_temp:
                    k = occupied_temp[q]
                    if k != j - 1 and k != j + 1 and k != i and sequence[k] == 'H':
                        delta += 1
            for q in _neighbours(new_j):
                if q in occupied_temp:
                    k = occupied_temp[q]
                    if k != j - 1 and k != j + 1 and k != i and sequence[k] == 'H':
                        delta -= 1
        
        new_energy = current_energy + delta
        
        if new_energy < current_energy:
            current_chain[i] = new_i
            current_chain[j] = new_j
            del occupied[old_i]
            del occupied[old_j]
            occupied[new_i] = i
            occupied[new_j] = j
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 200.0
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Generate templates
    templates = _generate_hamiltonian_templates(n, rng, num_templates=30)
    
    # Evaluate templates with greedy assignment
    for template in templates:
        if time.time() > deadline - 50.0:
            break
        
        assigned_chain, energy = _greedy_assign(sequence, template, rng)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(assigned_chain)
    
    # Local search on best template
    if best_chain is not None and time.time() < deadline - 30.0:
        refined_chain, refined_energy = _local_search(sequence, best_chain, rng, max_iter=3000, deadline=deadline - 25.0)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Two-point search
    if best_chain is not None and time.time() < deadline - 15.0:
        refined_chain, refined_energy = _two_point_search(sequence, best_chain, rng, max_iter=1000, deadline=deadline - 10.0)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Final polish
    if best_chain is not None and time.time() < deadline - 5.0:
        refined_chain, refined_energy = _local_search(sequence, best_chain, rng, max_iter=2000, deadline=deadline - 2.0)
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
