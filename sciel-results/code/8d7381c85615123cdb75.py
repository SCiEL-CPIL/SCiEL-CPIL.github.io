"""HP lattice protein folding on the square lattice using pivot local search with greedy polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 190.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            # Check 4 neighbors directly (faster than generator)
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                j = where.get((x+dx, y+dy))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_polish(sequence, chain, time_limit, rng):
    """Local search polish: try single-step moves and pinch moves that improve energy."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try moving each interior residue to an adjacent empty site
        for i in range(1, n - 1):
            if (time.time() - start_time) > time_limit:
                break
            x, y = current_chain[i]
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                # Check if new_pos is free (not in chain except at position i)
                if new_pos in current_chain:
                    continue
                # Check unit steps with neighbors
                if i > 0:
                    px, py = current_chain[i-1]
                    if abs(nx - px) + abs(ny - py) != 1:
                        continue
                if i < n - 1:
                    px, py = current_chain[i+1]
                    if abs(nx - px) + abs(ny - py) != 1:
                        continue
                
                # Try the move
                old_pos = current_chain[i]
                current_chain[i] = new_pos
                
                new_energy = _energy(sequence, current_chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                else:
                    current_chain[i] = old_pos
        
        # Pinch moves: directly target H-H contact formation
        for i in range(1, n - 1):
            if (time.time() - start_time) > time_limit:
                break
            if sequence[i] != 'H':
                continue
            x, y = current_chain[i]
            # Find nearby H residues that could form contacts
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                target_pos = (x + dx, y + dy)
                # Look for H residues within distance 2 that could be moved adjacent
                for j in range(1, n - 1):
                    if j == i or sequence[j] != 'H':
                        continue
                    jx, jy = current_chain[j]
                    # Only consider moves that bring j closer to i
                    if abs(jx - x) + abs(jy - y) > 3:
                        continue
                    for dx2, dy2 in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx2, ny2 = jx + dx2, jy + dy2
                        new_pos2 = (nx2, ny2)
                        if new_pos2 in current_chain:
                            continue
                        # Check unit steps with j's neighbors
                        if j > 0:
                            px, py = current_chain[j-1]
                            if abs(nx2 - px) + abs(ny2 - py) != 1:
                                continue
                        if j < n - 1:
                            px, py = current_chain[j+1]
                            if abs(nx2 - px) + abs(ny2 - py) != 1:
                                continue
                        
                        old_pos2 = current_chain[j]
                        current_chain[j] = new_pos2
                        new_energy = _energy(sequence, current_chain)
                        if new_energy < current_energy:
                            current_energy = new_energy
                            improved = True
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_chain = list(current_chain)
                        else:
                            current_chain[j] = old_pos2
                        
    return best_chain, best_energy

def _greedy_grow(sequence, dimension, rng):
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

def _try_pivot_move(sequence, chain, k, rot_choice):
    """
    Attempt a pivot move at index k with a specific rotation.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    px, py = chain[k]
    
    # Determine rotation using precomputed lookup for speed
    if rot_choice == 1:  # 90 deg
        def rot(ox, oy): return (-oy, ox)
    elif rot_choice == 2:  # 180 deg
        def rot(ox, oy): return (-ox, -oy)
    else:  # 270 deg
        def rot(ox, oy): return (oy, -ox)
        
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

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters - tuned for better exploration
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    # Scale iterations with available time
    max_iters = min(300000, int(time_limit * 5000))
    cooling_factor = (T_end / T_start) ** (1.0 / max(1, max_iters))
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
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    # Allocate remaining time, but leave a safety margin
    remaining = _TOTAL_BUDGET - elapsed
    # Scale time allocation by sequence length: shorter sequences need less time
    base_time = min(remaining, 20.0)
    # Shorter sequences converge faster
    if n <= 25:
        time_limit = base_time * 0.35
    elif n <= 40:
        time_limit = base_time * 0.5
    elif n <= 60:
        time_limit = base_time * 0.7
    elif n <= 80:
        time_limit = base_time * 0.85
    else:
        time_limit = base_time
    
    if time_limit <= 0.5:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on time budget - increase for better coverage
    if time_limit > 10.0:
        num_restarts = 5
    elif time_limit > 6.0:
        num_restarts = 4
    elif time_limit > 3.0:
        num_restarts = 3
    else:
        num_restarts = 2
        
    # Reserve 20% of time for final polish (pinch moves need more time)
    sa_time_per_restart = (time_limit * 0.80) / num_restarts
    polish_time = time_limit * 0.20
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine with pivot SA
        refined_chain, refined_energy = _pivot_search(sequence, initial, sa_time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    # Final greedy polish on the best chain found
    if best_chain is not None and polish_time > 0.1:
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, polish_time, rng)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
