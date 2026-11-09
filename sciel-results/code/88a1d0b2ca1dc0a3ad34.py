"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    # Each contact counted twice (once from each H), so divide by 2
    return -contacts // 2


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth with local search."""
    n = len(sequence)
    if n <= 2:
        return [(i, 0) for i in range(n)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    rng.shuffle(dirs)
    
    for i in range(1, n):
        # Try each direction in random order
        placed = False
        for dx, dy in dirs:
            x, y = chain[-1]
            new_pt = (x + dx, y + dy)
            if new_pt in occupied:
                continue
            
            # Compute gain: number of new H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                               (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                    q = (nx, ny)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Also check if the new position is adjacent to any H that will come later
            # (this is harder to predict, so we just use immediate gain)
            
            # Free space heuristic: prefer positions with more free neighbors
            free = 0
            for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                           (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                if (nx, ny) not in occupied:
                    free += 1
            
            # Accept this position
            occupied[new_pt] = i
            chain.append(new_pt)
            placed = True
            break
        
        if not placed:
            return None
    
    return chain


def _pivot_move(chain, pivot_idx, rng):
    """Attempt a pivot move: rotate the segment after pivot_idx around pivot_idx.
    Returns (success, new_chain) or (False, None)."""
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return False, None
    
    # Get the segment to rotate
    segment = chain[pivot_idx + 1:]
    pivot = chain[pivot_idx]
    before = chain[pivot_idx]
    after = chain[pivot_idx + 1]
    
    # Direction from pivot to next
    dx = after[0] - pivot[0]
    dy = after[1] - pivot[1]
    
    # Possible rotations: 90 degrees clockwise, 90 degrees counter-clockwise
    # Original direction: (dx, dy)
    # Clockwise: (dy, -dx)
    # Counter-clockwise: (-dy, dx)
    
    rotations = [(dy, -dx), (-dy, dx)]
    rng.shuffle(rotations)
    
    for new_dx, new_dy in rotations:
        # Generate new segment
        new_segment = []
        x, y = pivot[0] + new_dx, pivot[1] + new_dy
        # We need to track the relative positions
        rel_points = [(p[0] - pivot[0], p[1] - pivot[1]) for p in segment]
        
        # Apply rotation to each relative point
        # Rotation matrix for 90 deg CW: (x,y) -> (y, -x)
        # Rotation matrix for 90 deg CCW: (x,y) -> (-y, x)
        # Determine which rotation we're applying
        if (new_dx, new_dy) == (dy, -dx):
            # Clockwise
            new_rel = [(ry, -rx) for (rx, ry) in rel_points]
        else:
            # Counter-clockwise
            new_rel = [(-ry, rx) for (rx, ry) in rel_points]
        
        new_segment = [(pivot[0] + rx, pivot[1] + ry) for (rx, ry) in new_rel]
        
        # Check self-avoidance
        # Build occupied set for the new chain
        new_chain = chain[:pivot_idx + 1] + new_segment
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            continue
        
        # Check unit steps
        for i in range(len(new_chain) - 1):
            dxs = abs(new_chain[i+1][0] - new_chain[i][0])
            dys = abs(new_chain[i+1][1] - new_chain[i][1])
            if dxs + dys != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        return True, new_chain
    
    return False, None


def _anneal(sequence, initial_chain, budget_seconds, rng):
    """Simulated annealing with pivot moves."""
    n = len(sequence)
    if n <= 2:
        return initial_chain
    
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = initial_chain
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    no_improve_steps = 0
    max_no_improve = 500
    
    start_t = time.time()
    steps = 0
    
    while _remaining_time() > budget_seconds:
        if no_improve_steps > max_no_improve:
            break
        
        # Try a pivot move
        pivot_idx = rng.randint(1, n - 2)
        success, new_chain = _pivot_move(current_chain, pivot_idx, rng)
        
        if not success:
            no_improve_steps += 1
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_E = new_energy - current_energy
        
        if delta_E <= 0 or rng.random() < exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = current_chain
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        steps += 1
        if steps % 10000 == 0:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation found via multiple greedy growths + annealing."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate diverse initial conformations via greedy growth
    candidates = []
    num_greedy = min(50, max(10, n // 2))
    
    for _ in range(num_greedy):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Evaluate all candidates
    best_chain = candidates[0]
    best_energy = _energy(sequence, best_chain)
    
    for c in candidates[1:]:
        e = _energy(sequence, c)
        if e < best_energy:
            best_energy = e
            best_chain = c
    
    # Phase 2: Anneal the best candidate (and possibly others)
    time_per_anneal = max(0.5, _remaining_time() * 0.7 / max(1, min(3, len(candidates))))
    
    # Anneal the top few candidates
    candidates.sort(key=lambda c: _energy(sequence, c))
    num_to_anneal = min(3, len(candidates))
    
    for c in candidates[:num_to_anneal]:
        if _remaining_time() < 5.0:
            break
        budget = min(time_per_anneal, _remaining_time() - 3.0)
        if budget < 0.5:
            break
        
        annealed = _anneal(sequence, c, budget, rng)
        e = _energy(sequence, annealed)
        if e < best_energy:
            best_energy = e
            best_chain = annealed
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
