# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {(0,) * dimension: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None
        
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    
    return chain


def _pivot_move(chain, occupied_map, dimension, rng, pivot_idx, angle_sign=1):
    """
    Apply a pivot move: rotate the tail from pivot_idx+1 onward
    around the bond (pivot_idx-1, pivot_idx).
    Returns (new_chain, new_occupied_map) or (None, None) if invalid.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None, None
    
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    
    # Find bond axis
    v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if v[i] != 0:
            bond_axis = i
            break
    
    if bond_axis == -1:
        return None, None
    
    # Choose rotation axis (one of the two perpendicular axes)
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    r_axis = rng.choice(rot_axes)
    
    # Rotation function
    def rotate_point(point, center, axis, sign):
        p = list(point)
        c = list(center)
        for k in range(dimension):
            p[k] -= c[k]
        
        if axis == 0:  # Rotate around X
            if sign == 1:
                p[1], p[2] = -p[2], p[1]
            else:
                p[1], p[2] = p[2], -p[1]
        elif axis == 1:  # Rotate around Y
            if sign == 1:
                p[0], p[2] = p[2], -p[0]
            else:
                p[0], p[2] = -p[2], p[0]
        elif axis == 2:  # Rotate around Z
            if sign == 1:
                p[0], p[1] = -p[1], p[0]
            else:
                p[0], p[1] = p[1], -p[0]
        
        for k in range(dimension):
            p[k] += c[k]
        return tuple(p)
    
    # Check head (indices 0 to pivot_idx-1)
    head_set = set(chain[:pivot_idx])
    
    # Build new tail
    new_tail = []
    new_occ = set()
    
    for i in range(pivot_idx + 1, n):
        orig = chain[i]
        new_pt = rotate_point(orig, p_pivot, r_axis, angle_sign)
        
        if new_pt in head_set:
            return None, None
        if new_pt in new_occ:
            return None, None
        
        if new_tail:
            prev = new_tail[-1]
            dist = sum(abs(new_pt[k] - prev[k]) for k in range(dimension))
            if dist != 1:
                return None, None
        
        new_tail.append(new_pt)
        new_occ.add(new_pt)
    
    # Check first tail point is adjacent to pivot
    first_new = new_tail[0]
    dist = sum(abs(first_new[k] - p_pivot[k]) for k in range(dimension))
    if dist != 1:
        return None, None
    
    # Build new chain and occupied map
    new_chain = chain[:pivot_idx + 1] + new_tail
    new_occ_map = {chain[i]: i for i in range(pivot_idx + 1)}
    for i, pt in enumerate(new_tail):
        new_occ_map[pt] = pivot_idx + 1 + i
    
    return new_chain, new_occ_map


def _kink_move(chain, occupied_map, dimension, rng, kink_idx):
    """
    Apply a kink move: rotate a short segment (2-3 residues) by 90 degrees.
    Returns (new_chain, new_occupied_map) or (None, None) if invalid.
    """
    n = len(chain)
    if kink_idx < 0 or kink_idx >= n:
        return None, None
    
    # Choose segment length: 2 or 3
    seg_len = rng.choice([2, 3])
    if kink_idx + seg_len > n:
        return None, None
    
    # The hinge is at kink_idx - 1 and kink_idx
    if kink_idx < 1:
        return None, None
    
    p_before = chain[kink_idx - 1]
    p_hinge = chain[kink_idx]
    
    # Bond axis
    v = [p_hinge[i] - p_before[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if v[i] != 0:
            bond_axis = i
            break
    
    if bond_axis == -1:
        return None, None
    
    # Rotation axis
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    r_axis = rng.choice(rot_axes)
    sign = rng.choice([1, -1])
    
    def rotate_point(point, center, axis, sign):
        p = list(point)
        c = list(center)
        for k in range(dimension):
            p[k] -= c[k]
        
        if axis == 0:
            if sign == 1:
                p[1], p[2] = -p[2], p[1]
            else:
                p[1], p[2] = p[2], -p[1]
        elif axis == 1:
            if sign == 1:
                p[0], p[2] = p[2], -p[0]
            else:
                p[0], p[2] = -p[2], p[0]
        elif axis == 2:
            if sign == 1:
                p[0], p[1] = -p[1], p[0]
            else:
                p[0], p[1] = p[1], -p[0]
        
        for k in range(dimension):
            p[k] += c[k]
        return tuple(p)
    
    # Head: everything before kink_idx
    head_set = set(chain[:kink_idx])
    
    # New segment
    new_seg = []
    new_occ = set()
    
    for i in range(kink_idx, kink_idx + seg_len):
        orig = chain[i]
        new_pt = rotate_point(orig, p_hinge, r_axis, sign)
        
        if new_pt in head_set:
            return None, None
        if new_pt in new_occ:
            return None, None
        
        if new_seg:
            prev = new_seg[-1]
            dist = sum(abs(new_pt[k] - prev[k]) for k in range(dimension))
            if dist != 1:
                return None, None
        
        new_seg.append(new_pt)
        new_occ.add(new_pt)
    
    # Check adjacency: first new seg point must be adjacent to p_before? No, to chain[kink_idx-1]
    # Actually, the first new point replaces chain[kink_idx], so it must be adjacent to chain[kink_idx-1]
    first_new = new_seg[0]
    dist = sum(abs(first_new[k] - p_before[k]) for k in range(dimension))
    if dist != 1:
        return None, None
    
    # Check adjacency: last new seg point to next chain point (if exists)
    if kink_idx + seg_len < n:
        last_new = new_seg[-1]
        next_pt = chain[kink_idx + seg_len]
        dist = sum(abs(last_new[k] - next_pt[k]) for k in range(dimension))
        if dist != 1:
            return None, None
    
    # Build new chain
    new_chain = chain[:kink_idx] + new_seg + chain[kink_idx + seg_len:]
    
    # Build new occupied map
    new_occ_map = {}
    for i in range(len(new_chain)):
        new_occ_map[new_chain[i]] = i
    
    return new_chain, new_occ_map


def _crank_move(chain, occupied_map, dimension, rng, crank_idx):
    """
    Apply a crank move: rotate a single bond (2 residues) by 90 degrees.
    This is a special case of kink with seg_len=1 effectively.
    """
    n = len(chain)
    if crank_idx < 1 or crank_idx >= n - 1:
        return None, None
    
    p_before = chain[crank_idx - 1]
    p_hinge = chain[crank_idx]
    
    v = [p_hinge[i] - p_before[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if v[i] != 0:
            bond_axis = i
            break
    
    if bond_axis == -1:
        return None, None
    
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    r_axis = rng.choice(rot_axes)
    sign = rng.choice([1, -1])
    
    def rotate_point(point, center, axis, sign):
        p = list(point)
        c = list(center)
        for k in range(dimension):
            p[k] -= c[k]
        
        if axis == 0:
            if sign == 1:
                p[1], p[2] = -p[2], p[1]
            else:
                p[1], p[2] = p[2], -p[1]
        elif axis == 1:
            if sign == 1:
                p[0], p[2] = p[2], -p[0]
            else:
                p[0], p[2] = -p[2], p[0]
        elif axis == 2:
            if sign == 1:
                p[0], p[1] = -p[1], p[0]
            else:
                p[0], p[1] = p[1], -p[0]
        
        for k in range(dimension):
            p[k] += c[k]
        return tuple(p)
    
    # Head
    head_set = set(chain[:crank_idx])
    
    # Rotate just the point at crank_idx
    new_pt = rotate_point(chain[crank_idx], p_hinge, r_axis, sign)
    
    if new_pt in head_set:
        return None, None
    
    # Check adjacency to previous
    dist = sum(abs(new_pt[k] - p_before[k]) for k in range(dimension))
    if dist != 1:
        return None, None
    
    # Check adjacency to next
    if crank_idx + 1 < n:
        next_pt = chain[crank_idx + 1]
        dist = sum(abs(new_pt[k] - next_pt[k]) for k in range(dimension))
        if dist != 1:
            return None, None
    
    new_chain = chain[:crank_idx] + [new_pt] + chain[crank_idx + 1:]
    new_occ_map = {new_chain[i]: i for i in range(len(new_chain))}
    
    return new_chain, new_occ_map


def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Multi-start Simulated Annealing with Pivot, Kink, and Crank moves.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Number of restarts: scale with time available
    n_restarts = max(3, int(time_limit / 3.0))
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 0.5:
            break
        
        # Generate initial chain
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            # Try greedy growth as fallback
            chain = []
            occupied = {}
            start = (0,) * dimension
            chain.append(start)
            occupied[start] = 0
            for i in range(1, n):
                options = [s for s in _neighbours(chain[-1]) if s not in occupied]
                if not options:
                    chain = None
                    break
                site = rng.choice(options)
                chain.append(site)
                occupied[site] = i
            if chain is None:
                continue
        
        occupied_map = {chain[i]: i for i in range(n)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = list(chain)
            best_energy = current_energy
        
        # Simulated Annealing
        T = 1.0
        alpha = 0.95
        local_best_chain = list(chain)
        local_best_energy = current_energy
        
        steps_per_restart = 200
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 0.5:
                break
            
            # Choose move type
            r = rng.random()
            if r < 0.5:
                # Pivot
                pivot_idx = rng.randint(1, n - 2)
                sign = rng.choice([1, -1])
                new_chain, new_occ = _pivot_move(chain, occupied_map, dimension, rng, pivot_idx, sign)
            elif r < 0.8:
                # Kink
                kink_idx = rng.randint(0, n - 1)
                new_chain, new_occ = _kink_move(chain, occupied_map, dimension, rng, kink_idx)
            else:
                # Crank
                crank_idx = rng.randint(1, n - 2)
                new_chain, new_occ = _crank_move(chain, occupied_map, dimension, rng, crank_idx)
            
            if new_chain is None:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or (T > 0.01 and rng.random() < math.exp(-delta_e / T)):
                chain = new_chain
                occupied_map = new_occ
                current_energy = new_energy
                
                if current_energy < local_best_energy:
                    local_best_chain = list(chain)
                    local_best_energy = current_energy
            
            T *= alpha
        
        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_chain = local_best_chain
    
    if best_chain is None:
        best_chain = _random_walk(sequence, dimension, rng)
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Entry point: fold a sequence on the cubic lattice.
    """
    rng = random.Random(12345 + len(sequence))
    
    # Time budget per sequence
    # Evaluator gives 300s for all 26 sequences total, so ~11.5s per sequence on average.
    # Use 10s per sequence to be safe.
    time_limit = 10.0
    
    best_chain = _sa_fold(sequence, dimension, rng, time_limit)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
