"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _is_self_avoiding(chain):
    return len(set(chain)) == len(chain)

def _valid_chain(chain, dimension):
    if not chain or len(chain) != dimension:
        return False
    if not _is_self_avoiding(chain):
        return False
    for i in range(len(chain) - 1):
        diff = sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1]))
        if diff != 1:
            return False
    return True

def _generate_helix_conformation(n, dimension, sequence):
    """Generate a helical conformation that packs tightly."""
    if dimension != 3:
        return None
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Try different helix patterns
    patterns = [
        [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)],  # square spiral in xy
        [(1,0,0), (0,0,1), (-1,0,0), (0,0,-1)],  # square spiral in xz
        [(0,1,0), (0,0,1), (0,-1,0), (0,0,-1)],  # square spiral in yz
    ]
    
    best_chain = None
    best_energy = 0
    
    for pattern in patterns:
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0)}
        valid = True
        
        for i in range(1, n):
            step_idx = (i - 1) % len(pattern)
            dx, dy, dz = pattern[step_idx]
            
            # Try the step, and if blocked, try alternatives
            found = False
            for alt in range(len(pattern)):
                adx, ady, adz = pattern[alt]
                new_pos = (chain[-1][0] + adx, chain[-1][1] + ady, chain[-1][2] + adz)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    found = True
                    break
            
            if not found:
                valid = False
                break
        
        if valid and len(chain) == n:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    return best_chain

def _generate_random_walk(n, dimension, rng, sequence):
    """Generate a random self-avoiding walk."""
    if dimension != 3:
        return None
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for i in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        rng.shuffle(neighbors)
        
        found = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied.add(site)
                found = True
                break
        
        if not found:
            return None
    
    return chain

def _pivot_move(chain, pivot_idx, dimension):
    """Perform a pivot move at the given index."""
    if pivot_idx <= 0 or pivot_idx >= len(chain) - 1:
        return None
    
    head = chain[pivot_idx - 1]
    tail_segment = chain[pivot_idx:]
    
    # Try all 6 directions for the first bond
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        new_first = (head[0] + dx, head[1] + dy, head[2] + dz)
        
        if new_first in set(chain[:pivot_idx]):
            continue
        
        new_chain = chain[:pivot_idx] + [new_first]
        occupied = set(new_chain)
        
        valid = True
        for j in range(1, len(tail_segment)):
            prev = new_chain[-1]
            # The relative direction from previous to current in tail_segment
            orig_prev = tail_segment[j-1]
            orig_curr = tail_segment[j]
            dir_x = orig_curr[0] - orig_prev[0]
            dir_y = orig_curr[1] - orig_prev[1]
            dir_z = orig_curr[2] - orig_prev[2]
            
            new_pos = (prev[0] + dir_x, prev[1] + dir_y, prev[2] + dir_z)
            
            if new_pos in occupied:
                valid = False
                break
            
            new_chain.append(new_pos)
            occupied.add(new_pos)
        
        if valid:
            return new_chain
    
    return None

def _shift_move(chain, shift_idx, dimension, rng):
    """Perform a shift move: move a middle segment by one step."""
    if shift_idx <= 0 or shift_idx >= len(chain) - 1:
        return None
    
    # Choose a segment to shift
    seg_len = min(5, len(chain) // 4)
    if seg_len < 1:
        return None
    
    start = max(1, shift_idx - seg_len // 2)
    end = min(len(chain) - 1, start + seg_len)
    
    if end - start < 1:
        return None
    
    segment = chain[start:end+1]
    before = chain[:start]
    after = chain[end+1:]
    
    # Try shifting the segment in 6 directions
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        new_segment = [(x+dx, y+dy, z+dz) for x, y, z in segment]
        
        new_chain = before + new_segment + after
        
        if not _valid_chain(new_chain, dimension):
            continue
        
        # Check connectivity at junctions
        if start > 0:
            diff = sum(abs(a-b) for a, b in zip(before[-1], new_segment[0]))
            if diff != 1:
                continue
        if end < len(chain) - 1:
            diff = sum(abs(a-b) for a, b in zip(new_segment[-1], after[0]))
            if diff != 1:
                continue
        
        return new_chain
    
    return None

def _local_search(sequence, chain, dimension, rng, time_limit=1.0):
    """Local search using pivot and shift moves."""
    if chain is None:
        return None
    
    start_time = time.time()
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    n = len(current)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move
            pivot_idx = rng.randint(1, n - 2)
            new_chain = _pivot_move(current, pivot_idx, dimension)
        else:
            # Shift move
            shift_idx = rng.randint(1, n - 2)
            new_chain = _shift_move(current, shift_idx, dimension, rng)
        
        if new_chain is None:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            no_improve = 0
            
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best, best_energy

def _simulated_annealing(sequence, chain, dimension, rng, time_limit=2.0):
    """Simulated annealing with local moves."""
    if chain is None:
        return None
    
    start_time = time.time()
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    n = len(current)
    initial_temp = 5.0
    cooling_rate = 0.99
    
    temp = initial_temp
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit and temp > 0.01:
        move_type = rng.random()
        
        if move_type < 0.5:
            pivot_idx = rng.randint(1, n - 2)
            new_chain = _pivot_move(current, pivot_idx, dimension)
        else:
            shift_idx = rng.randint(1, n - 2)
            new_chain = _shift_move(current, shift_idx, dimension, rng)
        
        if new_chain is None:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current = new_chain
            current_energy = new_energy
            no_improve = 0
            
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        temp *= cooling_rate
    
    return best, best_energy

def fold(sequence, dimension):
    """Best conformation using multiple strategies with time management."""
    if dimension != 3:
        # Fallback for non-3D: straight line
        return [[i, 0, 0] for i in range(len(sequence))]
    
    n = len(sequence)
    global_deadline = time.time() + 200  # Leave 130s margin from 330s
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(42 + len(sequence))
    
    # Strategy 1: Helical conformations
    if time.time() < global_deadline - 100:
        helix_chain = _generate_helix_conformation(n, dimension, sequence)
        if helix_chain is not None:
            energy = _energy(sequence, helix_chain)
            if best_chain is None or energy < best_energy:
                best_chain = helix_chain
                best_energy = energy
    
    # Strategy 2: Multiple random walks + local search
    num_random_walks = 20
    for i in range(num_random_walks):
        if time.time() >= global_deadline - 50:
            break
        
        rw_chain = _generate_random_walk(n, dimension, rng, sequence)
        if rw_chain is None:
            continue
        
        # Local search on this chain
        ls_chain, ls_energy = _local_search(sequence, rw_chain, dimension, rng, time_limit=0.5)
        if ls_chain is not None:
            if best_chain is None or ls_energy < best_energy:
                best_chain = ls_chain
                best_energy = ls_energy
    
    # Strategy 3: Simulated annealing from best so far
    if time.time() < global_deadline - 30 and best_chain is not None:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, dimension, rng, time_limit=5.0)
        if sa_chain is not None:
            if best_chain is None or sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy
    
    # Strategy 4: More random walks if time permits
    if time.time() < global_deadline - 20:
        additional_walks = 30
        for i in range(additional_walks):
            if time.time() >= global_deadline - 10:
                break
            
            rw_chain = _generate_random_walk(n, dimension, rng, sequence)
            if rw_chain is None:
                continue
            
            ls_chain, ls_energy = _local_search(sequence, rw_chain, dimension, rng, time_limit=0.3)
            if ls_chain is not None:
                if best_chain is None or ls_energy < best_energy:
                    best_chain = ls_chain
                    best_energy = ls_energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
