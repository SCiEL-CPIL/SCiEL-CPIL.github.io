"""HP lattice protein folding: contact-guided construction + multi-start local search."""
import time
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
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
        for q in _neighbours(old_pos):
            if q in occupied:
                j = occupied[q]
                if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta += 1
        for q in _neighbours(new_pos):
            if q in occupied:
                j = occupied[q]
                if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta -= 1
    return delta

def _contact_guided_growth(sequence, rng, seed_pos=None):
    """Grow chain by explicitly targeting H-H contacts during construction."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    if seed_pos is None:
        seed_pos = (0, 0)
    
    chain = [seed_pos]
    occupied = {seed_pos: 0}
    placed_h = set()
    if sequence[0] == 'H':
        placed_h.add(0)
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            
            # Primary: direct H-H contact gain
            if sequence[i] == 'H':
                direct_contacts = 0
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            direct_contacts += 1
                score += direct_contacts * 15.0
            
            # Secondary: potential future contacts
            if sequence[i] == 'H':
                future_h_count = sum(1 for j in range(i + 2, n) if sequence[j] == 'H')
                if future_h_count > 0:
                    empty_neighbors = sum(1 for q in _neighbours(site) if q not in occupied)
                    score += empty_neighbors * 2.0
            
            # Tertiary: compactness
            if len(chain) > 3:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy)
                score -= dist * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Sort by score descending, pick with some randomness
        candidates.sort(key=lambda x: -x[0])
        
        # Pick from top candidates with softmax-like probability
        top_k = min(3, len(candidates))
        weights = [math.exp(candidates[k][0] / 3.0) for k in range(top_k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen_site = candidates[0][1]
        for k in range(top_k):
            cumulative += weights[k]
            if r <= cumulative:
                chosen_site = candidates[k][1]
                break
        
        chain.append(chosen_site)
        occupied[chosen_site] = i
        if sequence[i] == 'H':
            placed_h.add(i)
    
    return chain

def _spiral_template(n, direction_start=(1, 0)):
    """Generate a compact spiral-shaped self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = direction_start
    
    for i in range(1, n):
        found = False
        for step in range(4):
            x, y = chain[-1]
            dx, dy = direction
            new_pos = (x + dx, y + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
            
            direction = (dy, -dx)
        
        if not found:
            return None
    
    return chain

def _snake_template(n, rng):
    """Generate a snake pattern in a rectangle."""
    # Find rectangle dimensions close to sqrt(n)
    best_w, best_h = 1, n
    for w in range(2, int(math.sqrt(n)) + 2):
        h = (n + w - 1) // w
        if w * h >= n and abs(w * h - n) < abs(best_w * best_h - n):
            best_w, best_h = w, h
    
    w, h = best_w, best_h
    if w * h < n:
        h += 1
    
    chain = []
    occupied = set()
    x, y = 0, 0
    dx, dy = 1, 0
    
    for i in range(n):
        if (x, y) in occupied:
            break
        chain.append((x, y))
        occupied.add((x, y))
        
        # Try to continue in current direction
        next_x, next_y = x + dx, y + dy
        if 0 <= next_x < w and 0 <= next_y < h and (next_x, next_y) not in occupied:
            x, y = next_x, next_y
        else:
            # Turn
            if dx != 0:
                dx, dy = 0, 1
            else:
                dx, dy = 1, 0
            next_x, next_y = x + dx, y + dy
            if 0 <= next_x < w and 0 <= next_y < h and (next_x, next_y) not in occupied:
                x, y = next_x, next_y
            else:
                break
    
    if len(chain) == n:
        return chain
    return None

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
        
        new_i = old_j
        new_j = old_i
        
        if new_i == new_j:
            continue
        
        prev_i = current_chain[i - 1]
        next_i = current_chain[i + 1]
        prev_j = current_chain[j - 1]
        next_j = current_chain[j + 1]
        
        occupied_temp = dict(occupied)
        del occupied_temp[old_i]
        del occupied_temp[old_j]
        
        if new_i in occupied_temp or new_j in occupied_temp:
            continue
        
        if abs(new_i[0] - prev_i[0]) + abs(new_i[1] - prev_i[1]) != 1:
            continue
        if abs(new_i[0] - next_i[0]) + abs(new_i[1] - next_i[1]) != 1:
            continue
        if abs(new_j[0] - prev_j[0]) + abs(new_j[1] - prev_j[1]) != 1:
            continue
        if abs(new_j[0] - next_j[0]) + abs(new_j[1] - next_j[1]) != 1:
            continue
        
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
    
    # Generate multiple initial conformations using different strategies
    initial_chains = []
    
    # Strategy 1: Contact-guided growth (multiple random restarts)
    for _ in range(5):
        chain = _contact_guided_growth(sequence, rng)
        if chain and len(chain) == n:
            initial_chains.append(chain)
    
    # Strategy 2: Spiral templates
    for dir_idx in range(4):
        dirs = [(1,0), (0,1), (-1,0), (0,-1)]
        chain = _spiral_template(n, dirs[dir_idx])
        if chain and len(chain) == n:
            initial_chains.append(chain)
    
    # Strategy 3: Snake templates
    for _ in range(3):
        chain = _snake_template(n, rng)
        if chain and len(chain) == n:
            initial_chains.append(chain)
    
    # Evaluate all initial chains
    for chain in initial_chains:
        if time.time() > deadline - 50.0:
            break
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # If no valid chain was generated, use a fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Local search on best initial chain
    if time.time() < deadline - 30.0:
        refined_chain, refined_energy = _local_search(
            sequence, best_chain, rng, 
            max_iter=3000, 
            deadline=deadline - 25.0
        )
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Two-point search
    if time.time() < deadline - 15.0:
        refined_chain, refined_energy = _two_point_search(
            sequence, best_chain, rng,
            max_iter=1000,
            deadline=deadline - 10.0
        )
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Final polish
    if time.time() < deadline - 5.0:
        refined_chain, refined_energy = _local_search(
            sequence, best_chain, rng,
            max_iter=2000,
            deadline=deadline - 2.0
        )
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    return [list(p) for p in best_chain]
