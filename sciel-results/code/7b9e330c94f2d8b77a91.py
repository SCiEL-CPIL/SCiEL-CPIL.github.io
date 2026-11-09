"""HP lattice protein folding on the cubic lattice.
Uses contact-map-guided growth followed by pivot-move local search.
"""
import random
import time
import math

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = time.time() + 315.0


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
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _contact_map(sequence):
    """
    Estimate likely H-H contact pairs based on sequence proximity.
    Returns a set of (i, j) pairs that should be contacts in a good fold.
    """
    n = len(sequence)
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = set()
    # In HP chains, contacts tend to form between H residues that are
    # separated by roughly the "contact distance" in the sequence
    # For a compact fold, expect ~n/4 to n/3 contacts
    target_contacts = max(5, n // 5)
    
    # Prioritize pairs with sequence distance in [4, n//2]
    # and both are H
    candidates = []
    for idx_i, i in enumerate(h_positions):
        for j in h_positions[idx_i+1:]:
            dist = j - i
            if 4 <= dist <= n // 2:
                # Score: prefer intermediate distances (not too close, not too far)
                score = -abs(dist - n//4) + random.random() * 0.1
                candidates.append((score, i, j))
    
    candidates.sort(reverse=True)
    for _, i, j in candidates[:target_contacts * 2]:
        contacts.add((min(i,j), max(i,j)))
    
    return contacts


def _grow_with_contacts(sequence, dimension, rng, target_contacts=None):
    """
    Greedy chain growth that tries to realize target H-H contacts.
    """
    if target_contacts is None:
        target_contacts = _contact_map(sequence)
    
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    realized = set()
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Count new H-H contacts
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
                        # Bonus if this is a target contact
                        pair = (min(i, j), max(i, j))
                        if pair in target_contacts and pair not in realized:
                            gain += 2
                            realized.add(pair)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.05 * free + rng.random() * 0.5, site))
        
        if not options:
            return None, realized
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain, realized


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
    
    # The pivot is at chain[i-1], and we rotate around the axis from chain[i-1] to chain[i]
    pivot = chain[i-1]
    axis_start = chain[i]
    
    # The axis direction is from pivot to axis_start (unit vector)
    ax = axis_start[0] - pivot[0]
    ay = axis_start[1] - pivot[1]
    az = axis_start[2] - pivot[2]
    
    def rotate_90(p, pivot_pt, axis):
        """Rotate point p by 90 degrees around axis through pivot_pt."""
        ax, ay, az = axis
        px, py, pz = p[0] - pivot_pt[0], p[1] - pivot_pt[1], p[2] - pivot_pt[2]
        # Rotation matrix for 90 deg around axis (ax, ay, az)
        # For unit axis along x: (x,y,z) -> (x,-z,y)
        # For unit axis along y: (x,y,z) -> (z,y,-x)
        # For unit axis along z: (x,y,z) -> (-y,x,z)
        if ax == 1:
            nx, ny, nz = px, -pz, py
        elif ax == -1:
            nx, ny, nz = px, pz, -py
        elif ay == 1:
            nx, ny, nz = pz, py, -px
        elif ay == -1:
            nx, ny, nz = -pz, py, px
        elif az == 1:
            nx, ny, nz = -py, px, pz
        elif az == -1:
            nx, ny, nz = py, -px, pz
        else:
            return None
        return (pivot_pt[0] + nx, pivot_pt[1] + ny, pivot_pt[2] + nz)
    
    def rotate_180(p, pivot_pt, axis):
        """Rotate point p by 180 degrees around axis through pivot_pt."""
        ax, ay, az = axis
        px, py, pz = p[0] - pivot_pt[0], p[1] - pivot_pt[1], p[2] - pivot_pt[2]
        if ax != 0:
            nx, ny, nz = px, -py, -pz
        elif ay != 0:
            nx, ny, nz = -px, py, -pz
        else:
            nx, ny, nz = -px, -py, pz
        return (pivot_pt[0] + nx, pivot_pt[1] + ny, pivot_pt[2] + nz)
    
    def rotate_270(p, pivot_pt, axis):
        """Rotate point p by 270 degrees around axis through pivot_pt."""
        ax, ay, az = axis
        px, py, pz = p[0] - pivot_pt[0], p[1] - pivot_pt[1], p[2] - pivot_pt[2]
        if ax == 1:
            nx, ny, nz = px, pz, -py
        elif ax == -1:
            nx, ny, nz = px, -pz, py
        elif ay == 1:
            nx, ny, nz = -pz, py, px
        elif ay == -1:
            nx, ny, nz = pz, py, -px
        elif az == 1:
            nx, ny, nz = py, -px, pz
        elif az == -1:
            nx, ny, nz = -py, px, -pz
        else:
            return None
        return (pivot_pt[0] + nx, pivot_pt[1] + ny, pivot_pt[2] + nz)
    
    # Try all 3 rotations (90, 180, 270)
    rotations = [rotate_90, rotate_180, rotate_270]
    rng.shuffle(rotations)
    
    for rot_func in rotations:
        new_tail = []
        valid = True
        for p in tail:
            np = rot_func(p, pivot, (ax, ay, az))
            if np is None:
                valid = False
                break
            new_tail.append(np)
        
        if not valid:
            continue
        
        new_chain = prefix + new_tail
        
        # Check self-avoidance
        pos_set = set()
        for p in new_chain:
            if p in pos_set:
                valid = False
                break
            pos_set.add(p)
        
        if not valid:
            continue
        
        # Check unit steps
        for k in range(len(new_chain) - 1):
            a, b = new_chain[k], new_chain[k+1]
            if abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        return new_chain, new_energy
    
    return None, None


def _local_search_pivot(sequence, chain, rng, max_steps=1000):
    """
    Local search using pivot moves with simulated annealing.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temp = 1.0
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 10:
            break
        
        # Pick a pivot position
        i = rng.randint(2, n - 2)
        
        new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
        if new_chain is None:
            continue
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temp *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple contact-guided greedy growths.
    Phase 2: Local optimization via pivot moves.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Precompute contact map
    target_contacts = _contact_map(sequence)
    
    # Phase 1: Multiple contact-guided greedy growths
    time_remaining = _DEADLINE - time.time()
    base_growths = 300 if n < 50 else 150
    num_growths = max(30, min(600, int(base_growths * (time_remaining / 300.0))))
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 20:
            break
        
        chain, _ = _grow_with_contacts(sequence, dimension, rng, target_contacts)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line is always self-avoiding
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local optimization via pivot moves
    time_remaining = _DEADLINE - time.time()
    # Budget for local optimization
    max_local_steps = max(200, min(20000, int(time_remaining * 100)))
    
    current_chain, current_energy = _local_search_pivot(
        sequence, best_chain, rng, max_steps=max_local_steps
    )
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_chain = current_chain
    
    # Phase 3: Additional restarts if time remains
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 30:
        num_restarts = max(1, min(5, int(time_remaining / 15)))
        for _ in range(num_restarts):
            if time.time() > _DEADLINE - 15:
                break
            chain, _ = _grow_with_contacts(sequence, dimension, rng, target_contacts)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
                # Quick local search on new best
                if _DEADLINE - time.time() > 15:
                    new_chain, new_energy = _local_search_pivot(
                        sequence, best_chain, rng, max_steps=500
                    )
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = new_chain
    
    # Ensure we return the best chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
