import random
import time
import math

# Global time tracking
_TOTAL_START = time.time()
_HARD_DEADLINE = _TOTAL_START + 310.0  # 20s safety margin from 330s limit

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate the energy of a chain. Energy = -contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _random_saw(n, dimension, rng):
    """Generate a random self-avoiding walk."""
    if n == 0:
        return []
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    return chain

def _pivot_move(chain, k, axis, angle, dimension):
    """
    Perform a pivot move.
    Cut at k. Rotate segment k..N-1 around the bond k-1 -> k.
    Returns new chain or None if collision.
    """
    n = len(chain)
    if k <= 0 or k >= n:
        return None
    
    # Determine axis index
    axis_idx = -1
    for i in range(dimension):
        if axis[i] != 0:
            axis_idx = i
            break
            
    if axis_idx == -1:
        return None
        
    new_chain = list(chain)
    moving = chain[k+1:]
    
    def rotate_point(p, origin, axis_vec, angle):
        rel = tuple(p[i] - origin[i] for i in range(dimension))
        
        if axis_vec[0] != 0:
            x, y, z = rel
            if angle == 90:
                new_y, new_z = -z, y
            elif angle == 180:
                new_y, new_z = -y, -z
            elif angle == 270:
                new_y, new_z = z, -y
            else:
                return None
            new_rel = (x, new_y, new_z)
            
        elif axis_vec[1] != 0:
            x, y, z = rel
            if angle == 90:
                new_x, new_z = z, -x
            elif angle == 180:
                new_x, new_z = -x, -z
            elif angle == 270:
                new_x, new_z = -z, x
            else:
                return None
            new_rel = (new_x, y, new_z)
            
        elif axis_vec[2] != 0:
            x, y, z = rel
            if angle == 90:
                new_x, new_y = -y, x
            elif angle == 180:
                new_x, new_y = -x, -y
            elif angle == 270:
                new_x, new_y = y, -x
            else:
                return None
            new_rel = (new_x, new_y, z)
        else:
            return None
            
        new_p = tuple(origin[i] + new_rel[i] for i in range(dimension))
        return new_p

    new_moving = []
    for p in moving:
        new_p = rotate_point(p, chain[k-1], axis, angle)
        if new_p is None:
            return None
        new_moving.append(new_p)
        
    new_chain[k+1:] = new_moving
    
    occupied_fixed = {chain[i] for i in range(k)}
    for p in new_moving:
        if p in occupied_fixed:
            return None
            
    return new_chain

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    This often finds better initial structures than pure random SAW.
    """
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

def _local_search(sequence, chain, rng, deadline):
    """
    Fast hill-climbing using local moves (single residue relocation).
    Much faster than pivot moves for local optimization.
    """
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old = best[i]
            prev = best[i-1]
            next_pos = best[i+1]
            
            # Find valid candidate sites:
            # Must be neighbor of prev AND neighbor of next_pos
            # AND not occupied by any other residue (or itself)
            candidates = set()
            for site in _neighbours(prev):
                if site in _neighbours(next_pos):
                    if site not in occupied or occupied[site] == i:
                        if site != old:
                            candidates.add(site)
            
            if not candidates:
                continue
                
            for site in candidates:
                # Calculate energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Lost contacts
                    for q in _neighbours(old):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta -= 1
                    # Gained contacts
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if sequence[j] == 'H':
                                delta += 1
                
                if delta > 0: # Energy decreases (more negative)
                    # Update state
                    occupied[old] = -1
                    occupied[site] = i
                    best[i] = site
                    best_e -= delta
                    improved = True
                    break # Move to next i
        
        if not improved:
            # If no improvement in a full pass, try a few random perturbations (pivot) to escape local optima
            # But only if we have time
            if time.time() < deadline - 1.0:
                for _ in range(10):
                    k = rng.randint(1, n-1)
                    axis_vec = tuple(best[k][i] - best[k-1][i] for i in range(dimension))
                    angle = rng.choice([90, 180, 270])
                    new_c = _pivot_move(best, k, axis_vec, angle, dimension)
                    if new_c:
                        new_e = _energy(sequence, new_c)
                        if new_e < best_e:
                            best = new_c
                            best_e = new_e
                            occupied = {p: i for i, p in enumerate(best)}
                            improved = True
                            break
                if not improved:
                    break
            else:
                break
                
    return best, best_e

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
        
    rng = random.Random(12345 + len(sequence))
    
    # Time budget management
    # We have 26 sequences. Total budget ~330s.
    # Per sequence budget: ~12s.
    # We check global deadline to be safe.
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Initial Generation
    # Mix of random SAWs and Greedy Growth
    # Greedy growth is O(N^2) but finds better structures.
    # Random SAW is O(N^2) but diverse.
    
    # Try a few greedy growths
    for _ in range(3):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    # Try a few random SAWs
    for _ in range(3):
        chain = _random_saw(n, dimension, rng)
        if chain is not None:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
        
    # Phase 2: Optimization
    # Use local search (fast) to polish the best candidate.
    # Then use Simulated Annealing with Pivot Moves for global exploration if time remains.
    
    # Check global deadline
    if time.time() > _HARD_DEADLINE - 5.0:
        return [list(p) for p in best_chain]
        
    # Local Search Polish
    # Give it a significant chunk of time, e.g., 50% of remaining per-seq budget
    # But capped by global deadline
    local_deadline = min(time.time() + 5.0, _HARD_DEADLINE - 10.0)
    best_chain, best_energy = _local_search(sequence, best_chain, rng, local_deadline)
    
    if time.time() > _HARD_DEADLINE - 5.0:
        return [list(p) for p in best_chain]
        
    # Simulated Annealing
    # If time remains, run SA to try to escape local optima
    current_chain = list(best_chain)
    current_energy = best_energy
    
    T = 5.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < _HARD_DEADLINE - 2.0:
        # Pick a move
        k = rng.randint(1, n-1)
        axis_vec = tuple(current_chain[k][i] - current_chain[k-1][i] for i in range(dimension))
        angle = rng.choice([90, 180, 270])
        
        new_chain = _pivot_move(current_chain, k, axis_vec, angle, dimension)
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    return [list(p) for p in best_chain]
