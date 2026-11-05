"""HP lattice protein folding using pivot-move SA with contact-guided growth and CSP-inspired refinement."""
import time
import random
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
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

def _contact_guided_growth(sequence, rng):
    """Grow chain by explicitly targeting future H-H contacts with CSP-inspired forward checking."""
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
            
            # Primary: direct H-H contact gain
            if sequence[i] == 'H':
                direct_contacts = 0
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            direct_contacts += 1
                score += direct_contacts * 15.0
            
            # Secondary: potential future contacts (CSP forward checking)
            if sequence[i] == 'H':
                future_h_count = sum(1 for j in range(i + 2, n) if sequence[j] == 'H')
                if future_h_count > 0:
                    empty_neighbors = sum(1 for q in _neighbours(site) if q not in occupied)
                    score += empty_neighbors * 2.0
            
            # Compactness
            if len(chain) > 3:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist_to_center = abs(site[0] - cx) + abs(site[1] - cy)
                score -= dist_to_center * 0.5
            
            # Avoid dead ends
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            if free_count == 0 and i < n - 1:
                score -= 100.0
            elif free_count == 1 and i < n - 3:
                score -= 10.0
            
            score += rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        if rng.random() < 0.1 and len(candidates) > 1:
            idx = min(rng.randint(1, 2), len(candidates) - 1)
            best_site = candidates[idx][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
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

def _try_pivot_move(sequence, chain, k, rot_choice):
    """Attempt a pivot move at index k with a specific rotation."""
    n = len(chain)
    px, py = chain[k]
    
    if rot_choice == 1:
        def rot(x, y): return (-y, x)
    elif rot_choice == 2:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k])
    
    new_chain = list(chain[:k+1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
            
        new_chain.append(new_pos)
        
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _pivot_sa(sequence, chain, rng, time_limit):
    """Simulated annealing with pivot moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.05
    start_time = time.time()
    
    max_iters = 50000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        k = rng.randint(0, n - 2)
        rot_choice = rng.randint(1, 3)
        
        new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _local_search(sequence, chain, rng, max_iter=300):
    """Local search with single-residue moves."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
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

def _greedy_polish(sequence, chain, max_rounds=30):
    """Greedy polish using pivot moves."""
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        for k in range(n - 1):
            for rot_choice in range(1, 4):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
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
    deadline = start_time + 195.0
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Contact-guided growth
    for attempt in range(10):
        if time.time() > deadline - 30.0:
            break
        chain = _contact_guided_growth(sequence, rng)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Strategy 2: Spiral templates
    for start_dir in [(1,0), (0,1), (-1,0), (0,-1)]:
        if time.time() > deadline - 30.0:
            break
        chain = _spiral_template(n, start_dir)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Strategy 3: Pivot SA refinement
    if time.time() < deadline - 10.0:
        sa_time = min(60.0, deadline - time.time() - 10.0)
        sa_chain, sa_energy = _pivot_sa(sequence, best_chain, rng, sa_time)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Strategy 4: Local search polish
    if time.time() < deadline - 5.0:
        ls_chain, ls_energy = _local_search(sequence, best_chain, rng, max_iter=200)
        if ls_energy < best_energy:
            best_chain = ls_chain
            best_energy = ls_energy
    
    # Strategy 5: Greedy pivot polish
    if time.time() < deadline - 2.0:
        gp_chain, gp_energy = _greedy_polish(sequence, best_chain)
        if gp_energy < best_energy:
            best_chain = gp_chain
            best_energy = gp_energy
    
    # Final local search
    if time.time() < deadline - 1.0:
        ls_chain, ls_energy = _local_search(sequence, best_chain, rng, max_iter=100)
        if ls_energy < best_energy:
            best_chain = ls_chain
            best_energy = ls_energy
    
    return [list(p) for p in best_chain]
