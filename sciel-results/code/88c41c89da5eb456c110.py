"""
HP lattice protein folding on the cubic lattice.
Combines multi-start greedy growth with a simulated-annealing local search
using pivot moves and tail re-growth, plus a deadline guard.
"""
import random
import time

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = time.time() + 320.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            # Unroll for speed
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(sequence, chain, i, rng):
    """
    Try a pivot move at position i: rotate the tail starting at i around
    the bond from i-1 to i. Returns (new_chain, new_energy) or (None, None).
    """
    n = len(chain)
    if i < 2 or i >= n - 1:
        return None, None
    
    prefix = chain[:i]
    tail = chain[i:]
    
    pivot = chain[i-1]
    axis_start = chain[i]
    
    ax = axis_start[0] - pivot[0]
    ay = axis_start[1] - pivot[1]
    az = axis_start[2] - pivot[2]
    
    occupied_prefix = set(prefix)
    
    candidate_ends = []
    for site in _neighbours(pivot):
        if site == chain[i]:
            continue
        if site in occupied_prefix:
            continue
        sx = site[0] - pivot[0]
        sy = site[1] - pivot[1]
        sz = site[2] - pivot[2]
        dot = sx*ax + sy*ay + sz*az
        if dot == 0:
            candidate_ends.append((site, 90))
        elif dot == -1:
            candidate_ends.append((site, 180))
    
    if not candidate_ends:
        return None, None
    
    site, angle = candidate_ends[rng.randint(0, len(candidate_ends)-1)]
    
    new_tail = []
    for pt in tail:
        px = pt[0] - pivot[0]
        py = pt[1] - pivot[1]
        pz = pt[2] - pivot[2]
        
        if angle == 90:
            # Cross product u x v, then add u * (u . v)
            uxv_x = ay*pz - az*py
            uxv_y = az*px - ax*pz
            uxv_z = ax*py - ay*px
            udv = ax*px + ay*py + az*pz
            rx = uxv_x + ax*udv
            ry = uxv_y + ay*udv
            rz = uxv_z + az*udv
        elif angle == 180:
            # Reflection: -v + 2 * (u . v) * u
            udv = ax*px + ay*py + az*pz
            rx = -px + 2*ax*udv
            ry = -py + 2*ay*udv
            rz = -pz + 2*az*udv
        else:
            return None, None
        
        new_pt = (pivot[0] + rx, pivot[1] + ry, pivot[2] + rz)
        new_tail.append(new_pt)
    
    new_chain = prefix + new_tail
    
    # Check self-avoidance
    seen = set()
    for pt in new_chain:
        if pt in seen:
            return None, None
        seen.add(pt)
    
    # Check unit steps (pivot moves preserve unit steps by construction if rotation is valid, 
    # but the connection at the pivot must be checked implicitly by the candidate_ends logic)
    # Actually, the bond (i-1, i) is fixed to the new site. The rest of the tail is rigid.
    # So we only need to ensure the new tail doesn't collide with the prefix.
    # The self-avoidance check above covers collisions with prefix and within tail.
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def _regrow_tail(sequence, chain, i, rng):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    
    new_chain = list(prefix)
    
    for k in range(i, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None, None
        
        site = max(options)[1]
        occupied[site] = k
        new_chain.append(site)
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple randomized greedy growths.
    Phase 2: Simulated annealing local search with pivot moves and tail re-growth.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    # Allocate ~30% of time to Phase 1, 70% to Phase 2
    time_remaining = _DEADLINE - time.time()
    phase1_budget = time_remaining * 0.3
    phase1_deadline = time.time() + phase1_budget
    
    base_growths = 400 if n < 50 else 150
    # Estimate growth time: roughly n * constant.
    # We'll just run as many as we can before phase1_deadline
    num_growths = 0
    while time.time() < phase1_deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        num_growths += 1
        
        # Safety break to avoid infinite loop if time check is slow
        if num_growths > 5000:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Simulated annealing
    current_chain = list(best_chain)
    current_energy = best_energy
    
    time_remaining = _DEADLINE - time.time()
    
    # Temperature schedule: start high, decay geometrically
    t0 = max(1.0, -best_energy * 0.5)
    t_min = 0.1
    cooling_rate = 0.999
    
    temperature = t0
    
    # Estimate max steps based on time remaining
    # Assume each step takes ~0.1ms for small chains, ~1ms for large
    avg_step_time = 0.0005 if n < 50 else 0.002
    max_steps = int(time_remaining / avg_step_time)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose move type
        # 50% Pivot, 50% Regrow
        if rng.random() < 0.5:
            if n >= 4:
                i = rng.randint(2, n - 2)
                new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
            else:
                new_chain, new_energy = None, None
        else:
            # Regrow tail
            # Bias towards shorter tails (larger i) for efficiency, but allow deep regrow
            # For small chains, regrow is very effective.
            if n < 30:
                i = rng.randint(1, n - 1)
            else:
                # For larger chains, regrowing the whole tail is expensive.
                # Regrow last 10-30 residues
                tail_len = rng.randint(5, min(30, n - 1))
                i = n - tail_len
            
            new_chain, new_energy = _regrow_tail(sequence, current_chain, i, rng)
        
        if new_chain is None:
            continue
        
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0:
            accept = True
        else:
            if temperature > t_min:
                accept = rng.random() < pow(2.718281828, -delta_e / temperature)
            else:
                accept = False
        
        if accept:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        
        # Cool down
        temperature *= cooling_rate
    
    # Final validation and conversion
    result = [list(p) for p in best_chain]
    return result
