"""HP lattice protein folding using MIS-inspired contact selection with local refinement."""
import time
import random
import math
from collections import defaultdict

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

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

def _delta_energy(sequence, chain, i, new_pos, occupied):
    """Compute the change in energy if residue i moves from chain[i] to new_pos."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0
    
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                old_contacts += 1
    
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                new_contacts += 1
    
    return -(new_contacts - old_contacts)

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
            
            # Turn right
            direction = (dy, -dx)
        
        if not found:
            return None
    
    return chain

def _generate_spiral_templates(n, num_templates=50):
    """Pre-compute multiple compact spiral templates of length n."""
    templates = []
    seen = set()
    
    start_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for t in range(num_templates):
        rng = random.Random(1000 + t)
        
        # Vary the spiral pattern slightly
        start_dir = start_dirs[t % 4]
        
        # Sometimes use a different turning pattern
        turn_pattern = t % 3  # 0: standard, 1: alternate, 2: random-ish
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        direction = start_dir
        
        for i in range(1, n):
            found = False
            
            if turn_pattern == 0:
                # Standard spiral
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
            elif turn_pattern == 1:
                # Alternate turning
                for step in range(4):
                    x, y = chain[-1]
                    dx, dy = direction
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        found = True
                        break
                    # Alternate between left and right turns
                    if step % 2 == 0:
                        direction = (dy, -dx)  # right
                    else:
                        direction = (-dy, dx)  # left
            else:
                # Random-ish with preference for compactness
                candidates = []
                x, y = chain[-1]
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        # Prefer positions that are closer to the centroid
                        cx = sum(p[0] for p in chain) / len(chain)
                        cy = sum(p[1] for p in chain) / len(chain)
                        dist = abs(new_pos[0] - cx) + abs(new_pos[1] - cy)
                        candidates.append((dist, new_pos))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    # Pick from top 2 with some randomness
                    if len(candidates) > 1 and rng.random() < 0.3:
                        new_pos = candidates[1][1]
                    else:
                        new_pos = candidates[0][1]
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    found = True
                else:
                    break
            
            if not found:
                break
        
        if len(chain) == n:
            # Normalize to start at origin
            min_x = min(p[0] for p in chain)
            min_y = min(p[1] for p in chain)
            normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
            if normalized not in seen:
                seen.add(normalized)
                templates.append(list(chain))
    
    return templates

def _greedy_contact_growth(sequence, rng, max_iter=5000):
    """Grow a chain using greedy contact maximization with MIS-inspired scoring."""
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
            
            # MIS-inspired: score based on potential H-H contacts
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            # Weight by how "valuable" this contact is
                            # Contacts with H residues that have few other contacts are more valuable
                            h_neighbors = sum(1 for r in _neighbours(q) if r in occupied and occupied[r] != j and sequence[occupied[r]] == 'H' and abs(occupied[r] - j) > 1)
                            score += 10.0 / (1 + h_neighbors * 0.5)
            
            # Prefer compact positions
            if len(chain) > 5:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist_to_center = abs(site[0] - cx) + abs(site[1] - cy)
                score -= dist_to_center * 0.3
            
            # Avoid dead ends
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            if free_count == 0 and i < n - 1:
                score -= 50.0
            elif free_count == 1 and i < n - 3:
                score -= 5.0
            
            # Add small random perturbation
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        # Sometimes pick a non-best candidate for diversity
        if rng.random() < 0.15 and len(candidates) > 1:
            idx = min(rng.randint(1, 3), len(candidates) - 1)
            best_site = candidates[idx][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_search(sequence, chain, rng, max_iter=1000):
    """Local search: try moving each interior residue to an adjacent empty position."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        # Try moving residues in random order
        order = list(range(1, n - 1))
        rng.shuffle(order)
        
        for i in order:
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                delta_e = _delta_energy(sequence, best_chain, i, new_pos, occupied)
                
                if delta_e < 0:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    best_chain[i] = new_pos
                    best_energy += delta_e
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, rng, start_temp=2.0, end_temp=0.05, max_iter=3000):
    """Simulated annealing to escape local minima."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    temp = start_temp
    cooling = (math.log(start_temp / end_temp) / max_iter) if max_iter > 0 else 0
    
    for _ in range(max_iter):
        temp *= math.exp(-cooling)
        if temp < end_temp:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        moves = []
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
            dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
            
            if dist_prev == 1 and dist_next == 1:
                moves.append(new_pos)
        
        if not moves:
            continue
        
        new_pos = moves[rng.randint(0, len(moves) - 1)]
        delta_e = _delta_energy(sequence, current_chain, i, new_pos, occupied)
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            del occupied[old_pos]
            occupied[new_pos] = i
            current_chain[i] = new_pos
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _hill_climb_polish(sequence, chain, rng, max_iter=200):
    """Aggressive hill-climbing polish pass."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        for i in range(1, n - 1):
            old_pos = current_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                delta_e = _delta_energy(sequence, current_chain, i, new_pos, occupied)
                
                if delta_e < 0:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta_e
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Use pre-computed compact spiral templates
    templates = _generate_spiral_templates(n, num_templates=40)
    
    for template in templates:
        if time.time() > deadline - 60.0:
            break
        
        energy = _energy(sequence, template)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(template)
        
        # Also try reversed
        rev_energy = _energy(sequence[::-1], template)
        if rev_energy < best_energy:
            best_energy = rev_energy
            best_chain = list(template)
    
    # Strategy 2: Multiple MIS-inspired greedy constructions
    num_attempts = 0
    max_attempts = 8000
    
    while num_attempts < max_attempts:
        if time.time() > deadline - 60.0:
            break
        
        strategy = rng.randint(0, 2)
        
        if strategy == 0:
            chain = _greedy_contact_growth(sequence, rng)
        elif strategy == 1:
            # Spiral with random start
            start_dir = [(1,0), (0,1), (-1,0), (0,-1)][rng.randint(0, 3)]
            chain = _spiral_template(n, start_dir)
        else:
            # Random walk with compactness bias
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            for i in range(1, n):
                last = chain[-1]
                candidates = []
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    cx = sum(p[0] for p in chain) / len(chain)
                    cy = sum(p[1] for p in chain) / len(chain)
                    dist = abs(site[0] - cx) + abs(site[1] - cy)
                    candidates.append((dist, site))
                if not candidates:
                    chain = None
                    break
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 1 and rng.random() < 0.2:
                    new_pos = candidates[rng.randint(1, min(2, len(candidates)-1))][1]
                else:
                    new_pos = candidates[0][1]
                chain.append(new_pos)
                occupied[new_pos] = i
        
        if chain is not None and len(chain) == n:
            num_attempts += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Strategy 3: Local refinement
    if best_chain is not None and time.time() < deadline - 20.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iter=500)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Strategy 4: Simulated annealing
    if best_chain is not None and time.time() < deadline - 10.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, start_temp=2.0, end_temp=0.05, max_iter=2000)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Strategy 5: Hill-climb polish
    if best_chain is not None and time.time() < deadline - 2.0:
        polish_chain, polish_energy = _hill_climb_polish(sequence, best_chain, rng, max_iter=300)
        if polish_energy < best_energy:
            best_chain = polish_chain
            best_energy = polish_energy
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
